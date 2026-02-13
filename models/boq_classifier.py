#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BoQ-style Classifier — 方案B
- 固定数据划分（random_state=42），其余保持随机（每次运行不同，可选固定）
- ResNet18 backbone + BoQ (Bag-of-Queries) 聚合 + MLP 分类头
"""

import os
import sys
import random
import argparse
import numpy as np
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast


# =============================
# 参数
# =============================
def parse_args():
    p = argparse.ArgumentParser("BoQ-style Classifier (ResNet18+BoQ)")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录（子文件夹为类别）")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入图像尺寸")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers (Windows 建议 0)")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张样本进行实验，0 表示不限制")

    # 训练
    p.add_argument("--batch_size", type=int, default=32,
                   help="batch size")
    p.add_argument("--epochs", type=int, default=10,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="权重衰减")
    p.add_argument("--use_pretrained", action="store_true",
                   help="使用 ImageNet 预训练 ResNet18")
    p.add_argument("--use_amp", action="store_true",
                   help="启用混合精度训练 (仅在 cuda 有效)")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前 K 个 epoch 冻结 backbone 只训 BoQ+head，0 表示不冻结")

    # BoQ 超参
    p.add_argument("--proj_dim", type=int, default=256,
                   help="ResNet 输出 512 -> proj_dim 的降维维度")
    p.add_argument("--num_queries", type=int, default=16,
                   help="BoQ 查询/码字数 K")
    p.add_argument("--drop_rate", type=float, default=0.2,
                   help="分类头 dropout")

    # 随机、设备、输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="设备选择")
    p.add_argument("--seed", type=int, default=-1,
                   help="全局随机种子；<0 表示每次随机 (方案B 原设定)")
    p.add_argument("--split_seed", type=int, default=42,
                   help="划分 train/test 的随机种子 (固定)")
    p.add_argument("--save_model", type=str, default="boq_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="若非空，保存混淆矩阵图到该路径，而不是直接 plt.show()")

    return p.parse_args()


# =============================
# 工具函数
# =============================
def select_device(arg: str):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def set_seed_dynamic_or_fixed(seed_arg: int):
    """
    方案B：
    - seed_arg < 0: 每次运行用新的随机种子（训练不复现）
    - seed_arg >= 0: 固定种子，方便对比实验
    """
    if seed_arg is None or seed_arg < 0:
        s = int.from_bytes(os.urandom(8), "little")
    else:
        s = int(seed_arg)

    np_seed = int(s % (2**32 - 1)) or 1
    th_seed = int(s % (2**63 - 1)) or 1

    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.manual_seed(th_seed)
    torch.cuda.manual_seed_all(th_seed)

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    print(f"[INFO] Seeds -> numpy:{np_seed} torch:{th_seed}", file=sys.stderr)
    return np_seed, th_seed


# =============================
# BoQ 聚合器
# =============================
class BoQ(nn.Module):
    """
    Learnable Queries 聚合:
      - 输入 x: [B, D, H, W]
      - 学习 K×D 的查询矩阵 Q，与 x 的空间向量点积 -> softmax 注意力 -> 加权和
      - 输出: [B, K*D] (L2 归一化)
    """
    def __init__(
        self,
        dim: int,
        num_queries: int = 16,
        normalize_input: bool = True,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.D = dim
        self.K = num_queries
        self.normalize_input = normalize_input

        self.queries = nn.Parameter(torch.randn(self.K, self.D))
        nn.init.normal_(self.queries, std=0.02)

        init_log_scale = np.log(10.0)  # 初始缩放 ~10
        self.logit_scale = nn.Parameter(
            torch.tensor(init_log_scale, dtype=torch.float32),
            requires_grad=learnable_temp,
        )

        # 每个 query 的门控
        self.gates = nn.Parameter(torch.zeros(self.K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        assert D == self.D, f"Channel mismatch: got {D}, expect {self.D}"
        N = H * W

        x_flat = x.view(B, D, N)  # [B,D,N]

        if self.normalize_input:
            x_n = F.normalize(x_flat, p=2, dim=1)
            q_n = F.normalize(self.queries, p=2, dim=1)
        else:
            x_n = x_flat
            q_n = self.queries

        # 注意力打分 [B,K,N]
        logits = torch.einsum("kd,bdn->bkn", q_n, x_n)
        scale = self.logit_scale.exp()
        attn = torch.softmax(logits * scale, dim=2)

        # 聚合 [B,K,D]
        desc = torch.einsum("bkn,bdn->bkd", attn, x_flat)

        # 门控: 初期在 0.5~1 之间，避免一开始全关
        g = torch.sigmoid(self.gates).view(1, self.K, 1)
        desc = desc * (0.5 + 0.5 * g)

        # 归一化 + 展平
        desc = F.normalize(desc, p=2, dim=2)  # 局部
        v = desc.reshape(B, -1)               # [B, K*D]
        v = F.normalize(v, p=2, dim=1)        # 全局
        return v


# =============================
# 模型
# =============================
class BoQClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        proj_dim: int = 256,
        K: int = 16,
        pretrained: bool = False,
        drop: float = 0.2,
    ):
        super().__init__()
        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # backbone: 到 layer4 输出 [B,512,H,W]
        self.backbone = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4,
        )

        self.reduce = nn.Conv2d(512, proj_dim, kernel_size=1, bias=False)
        self.bn_red = nn.BatchNorm2d(proj_dim)
        self.relu = nn.ReLU(inplace=True)

        self.boq = BoQ(dim=proj_dim, num_queries=K, normalize_input=True, learnable_temp=True)
        feat_dim = K * proj_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        fm = self.backbone(x)                       # [B,512,H,W]
        fm = self.relu(self.bn_red(self.reduce(fm)))# [B,proj_dim,H,W]
        v = self.boq(fm)                            # [B,K*proj_dim]
        logits = self.head(v)
        return logits


# =============================
# 训练 / 验证
# =============================
def set_backbone_trainable(model: BoQClassifier, flag: bool):
    for p in model.backbone.parameters():
        p.requires_grad = flag


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scaler: GradScaler = None,
    clip_grad_norm: float = 0.0,
    train: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []
    seen = 0
    t0 = perf_counter()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            if scaler is not None and scaler.is_enabled():
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )
                    optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        seen += bs

    dt = perf_counter() - t0
    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1, dt, seen


# =============================
# 主流程
# =============================
def main(args):
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    set_seed_dynamic_or_fixed(args.seed)

    # ---------- 数据 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_set = datasets.ImageFolder(root=args.data_dir, transform=transform)
    num_classes = len(full_set.classes)

    if num_classes <= 1:
        raise RuntimeError("类别数 <= 1，请检查数据集目录结构（需要多类别子文件夹）。")

    indices = np.arange(len(full_set))
    labels = np.array(full_set.targets)

    # 可选限制样本总数（从前面截取，保持 stratify 之前的顺序）
    if args.limit and args.limit < len(indices):
        indices = indices[: args.limit]
        labels = labels[: args.limit]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=labels,
    )

    train_set = Subset(full_set, train_idx)
    test_set = Subset(full_set, test_idx)

    # DataLoader shuffle 使用独立随机源（保持“其余随机”）
    dl_rng = torch.Generator()
    dl_rng.manual_seed(
        int.from_bytes(os.urandom(8), "little") % (2**63 - 1) or 1
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=dl_rng,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Classes ({num_classes}): {full_set.classes}", file=sys.stderr)
    print(f"[INFO] Train: {len(train_set)} | Test: {len(test_set)}", file=sys.stderr)

    # ---------- 模型 ----------
    model = BoQClassifier(
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        K=args.num_queries,
        pretrained=args.use_pretrained,
        drop=args.drop_rate,
    ).to(device)

    print(model, file=sys.stderr)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda" and args.use_amp))

    # ---------- 训练 ----------
    best_f1 = 0.0
    best_path = args.save_model

    for epoch in range(1, args.epochs + 1):
        # 可选冻结 backbone
        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        tr = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad_norm=0.0,
            train=True,
        )
        scheduler.step()
        te = run_epoch(
            model,
            test_loader,
            device,
            criterion,
            optimizer=None,
            scaler=None,
            clip_grad_norm=0.0,
            train=False,
        )

        train_loss, _, _, _, _, tr_dt, tr_seen = tr
        (
            test_loss,
            test_acc,
            test_prec,
            test_rec,
            test_f1,
            te_dt,
            te_seen,
        ) = te

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} | "
            f"Test acc {test_acc:.4f}  "
            f"P {test_prec:.4f}  R {test_rec:.4f}  F1 {test_f1:.4f} | "
            f"Train {tr_seen / max(tr_dt,1e-6):.1f} img/s, "
            f"Test {te_seen / max(te_dt,1e-6):.1f} img/s"
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), best_path)
            print(
                f"[INFO] New best F1={best_f1:.4f}, model saved to {best_path}",
                file=sys.stderr,
            )

    # ---------- 最终评估 + 混淆矩阵 ----------
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}", file=sys.stderr)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    print(f"[INFO] Best macro-F1: {best_f1:.4f}")

    if args.save_cm:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (BoQ-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(args.save_cm, dpi=150)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {args.save_cm}", file=sys.stderr)
    else:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (BoQ-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

