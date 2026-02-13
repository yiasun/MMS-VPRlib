#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MixVPR-style Classifier (CLI Version)

Backbone:
  - ResNet18 至 layer4

Aggregator:
  - 1x1 Conv 将 512 -> proj_dim
  - 多个 MixBlock: (1x1 conv -> GELU -> depthwise conv -> GELU -> 1x1 conv) 残差结构

Pooling:
  - GeM (learnable p)

Head:
  - BN + Dropout + Linear

Metrics:
  - Accuracy, macro Precision / Recall / F1, Confusion Matrix
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
# 0) 参数
# =============================
def parse_args():
    p = argparse.ArgumentParser("MixVPR-style Classifier")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录（子文件夹为类别）")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入尺寸，默认 224")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers，Windows 建议 0")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图做实验，0 为不限制")

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
                   help="启用混合精度（需 CUDA）")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前 K 个 epoch 冻结 backbone 只训聚合器+头")

    # MixVPR 聚合器超参
    p.add_argument("--proj_dim", type=int, default=256,
                   help="通道降维维度 (512->proj_dim)")
    p.add_argument("--num_mix_blocks", type=int, default=3,
                   help="MixBlock 层数")
    p.add_argument("--dw_kernel", type=int, default=7,
                   help="depthwise 卷积核大小（奇数较好）")
    p.add_argument("--mlp_ratio", type=float, default=2.0,
                   help="MixBlock 内隐藏通道扩展倍数")
    p.add_argument("--drop_rate", type=float, default=0.1,
                   help="MixBlock 内 dropout")

    # 设备 / 随机数 / 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="运行设备")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（用于数据划分等）")
    p.add_argument("--split_seed", type=int, default=42,
                   help="train/test 划分种子")
    p.add_argument("--save_model", type=str, default="mixvpr_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="混淆矩阵保存路径；留空则直接展示")

    return p.parse_args()


def select_device(arg: str):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    print(f"[INFO] Global seed set to {seed}", file=sys.stderr)


# =============================
# 2) GeM 池化
# =============================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learnable)
        self.eps = eps

    def forward(self, x):  # x: [B,C,H,W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / self.p)
        return x.view(x.size(0), -1)  # [B,C]


# =============================
# 3) MixVPR-style 聚合块
# =============================
class MixBlock(nn.Module):
    """
    x -> 1x1 conv (expand) -> BN -> GELU
      -> depthwise conv(kxk) -> BN -> GELU
      -> 1x1 conv (project) -> BN -> Dropout
      -> 残差 + ReLU
    """
    def __init__(self, dim, mlp_ratio=2.0, dw_kernel=7, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=hidden,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden)

        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        idt = x
        y = self.pw1(x)
        y = self.bn1(y)
        y = F.gelu(y)

        y = self.dw(y)
        y = self.bn2(y)
        y = F.gelu(y)

        y = self.pw2(y)
        y = self.bn3(y)
        y = self.drop(y)

        return F.relu(idt + y, inplace=True)


# =============================
# 4) 模型定义
# =============================
class MixVPRClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        proj_dim=256,
        num_mix=3,
        dw_kernel=7,
        mlp_ratio=2.0,
        drop=0.1,
        pretrained=False,
    ):
        super().__init__()

        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4,
        )  # [B,512,H,W]

        # 1x1 降维到 proj_dim
        self.reduce = nn.Conv2d(512, proj_dim, kernel_size=1, bias=False)
        self.bn_red = nn.BatchNorm2d(proj_dim)
        self.relu = nn.ReLU(inplace=True)

        # 多层 MixBlock
        self.mix = nn.Sequential(
            *[
                MixBlock(
                    proj_dim,
                    mlp_ratio=mlp_ratio,
                    dw_kernel=dw_kernel,
                    drop=drop,
                )
                for _ in range(num_mix)
            ]
        )

        # GeM Pooling
        self.pool = GeM(p=3.0, learnable=True)  # [B,proj_dim]

        self.bn_vec = nn.BatchNorm1d(proj_dim)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)                        # [B,512,H,W]
        x = self.relu(self.bn_red(self.reduce(x)))  # [B,proj_dim,H,W]
        x = self.mix(x)                             # [B,proj_dim,H,W]
        v = self.pool(x)                            # [B,proj_dim]
        v = self.bn_vec(v)
        logits = self.head(v)                       # [B,num_classes]
        return logits


# =============================
# 5) 训练与评测
# =============================
def set_backbone_trainable(model: MixVPRClassifier, flag: bool):
    for p in model.backbone.parameters():
        p.requires_grad = flag


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scaler: GradScaler = None,
    train: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []
    t0 = perf_counter()
    seen = 0

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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
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


def main(args):
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    set_seed_all(args.seed)

    # ========= 数据 =========
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
        raise RuntimeError("类别数 <= 1，请检查数据集（子文件夹=类别）。")

    indices = np.arange(len(full_set))
    labels = np.array(full_set.targets)

    # 可选：限制样本数以便快速实验
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

    # DataLoader（训练 shuffle 使用独立随机源）
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

    # ========= 模型 =========
    model = MixVPRClassifier(
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        num_mix=args.num_mix_blocks,
        dw_kernel=args.dw_kernel,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop_rate,
        pretrained=args.use_pretrained,
    ).to(device)
    print(model, file=sys.stderr)

    # ========= 优化器等 =========
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

    best_f1 = 0.0
    best_path = args.save_model

    # ========= 训练循环 =========
    for epoch in range(1, args.epochs + 1):
        # 可选：前若干轮冻结 backbone
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
            train=False,
        )

        train_loss, _, _, _, _, tr_dt, tr_seen = tr
        _, te_acc, te_prec, te_rec, te_f1, te_dt, te_seen = te

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} | "
            f"Test acc {te_acc:.4f}  "
            f"P {te_prec:.4f}  R {te_rec:.4f}  F1 {te_f1:.4f} | "
            f"Train {tr_seen / max(tr_dt,1e-6):.1f} img/s, "
            f"Test {te_seen / max(te_dt,1e-6):.1f} img/s"
        )

        if te_f1 > best_f1:
            best_f1 = te_f1
            torch.save(model.state_dict(), best_path)
            print(
                f"[INFO] New best F1={best_f1:.4f}, model saved to {best_path}",
                file=sys.stderr,
            )

    # ========= 最终评估 + 混淆矩阵 =========
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
        plt.title("Confusion Matrix (MixVPR-style Classifier)")
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
        plt.title("Confusion Matrix (MixVPR-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
