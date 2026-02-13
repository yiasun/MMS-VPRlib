#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch-NetVLAD style classifier (CLI)

Backbone:
  - ResNet18 到 layer4

Projection:
  - 1x1 Conv: 512 -> proj_dim (默认 128)

Multi-scale Regions:
  - grids [1,2,3] -> 总区域数 = 1^2 + 2^2 + 3^2 = 14
  - 所有区域共享同一套 NetVLAD (clusters + assignment conv)

Head:
  - 拼接所有区域的 NetVLAD 描述 [B, R * K * D] -> MLP 分类

Metrics:
  - Accuracy, macro Precision / Recall / F1, Confusion Matrix
"""

import os
import sys
import argparse
import random
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


# =========================
# 0) 参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("Patch-NetVLAD style classifier")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录 (子文件夹为类别)")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入图像尺寸，默认 224")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers，Windows 建议 0")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图片做实验，0 表示不限制")

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
                   help="启用混合精度 (需要 CUDA)")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前 K 个 epoch 冻结 backbone，只训练 NetVLAD+head")

    # Patch-NetVLAD 超参
    p.add_argument("--vlad_K", type=int, default=16,
                   help="NetVLAD 聚类中心数 K")
    p.add_argument("--proj_dim", type=int, default=128,
                   help="1x1 conv 降维后的通道数")
    p.add_argument("--grids", type=int, nargs="+", default=[1, 2, 3],
                   help="多尺度网格划分，例如: --grids 1 2 3")

    # 设备 / 随机 / 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="运行设备")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（用于划分和复现实验）")
    p.add_argument("--split_seed", type=int, default=42,
                   help="train/test 划分种子")
    p.add_argument("--save_model", type=str, default="patch_netvlad_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="混淆矩阵保存路径，为空则直接展示")

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
    print(f"[INFO] Global seed = {seed}", file=sys.stderr)


# =========================
# 1) NetVLAD 模块
# =========================
class NetVLAD(nn.Module):
    """
    输入: x ∈ [B, D, H, W]
    输出: v ∈ [B, K*D]
    典型 NetVLAD：soft assignment + residual sum + intra / global L2
    """
    def __init__(self, num_clusters=16, dim=128, normalize_input=True):
        super().__init__()
        self.K = num_clusters
        self.D = dim
        self.normalize_input = normalize_input

        self.clusters = nn.Parameter(torch.randn(self.K, self.D))
        self.assignment = nn.Conv2d(self.D, self.K, kernel_size=1, bias=True)
        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.assignment.weight, nonlinearity="linear")
        nn.init.constant_(self.assignment.bias, 0.0)
        nn.init.normal_(self.clusters, std=1.0)

    def forward(self, x):
        """
        x: [B, D, H, W]
        """
        B, D, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft assignment
        soft_assign = F.softmax(self.assignment(x), dim=1)   # [B, K, H, W]
        N = H * W

        x_flat = x.view(B, D, N)                             # [B, D, N]
        a_flat = soft_assign.view(B, self.K, N)              # [B, K, N]

        # residual aggregation
        term1 = torch.einsum("bkn,bdn->bkd", a_flat, x_flat)             # [B, K, D]
        a_sum = a_flat.sum(dim=2)                                       # [B, K]
        term2 = a_sum.unsqueeze(2) * self.clusters.unsqueeze(0)         # [B, K, D]

        vlad = term1 - term2                                            # [B, K, D]
        vlad = F.normalize(vlad, p=2, dim=2)                            # intra-normalize
        vlad = vlad.reshape(B, -1)                                      # [B, K*D]
        vlad = F.normalize(vlad, p=2, dim=1)                            # global-normalize
        return vlad


# =========================
# 2) 区域划分工具
# =========================
def grid_bounds(S: int, g: int):
    """
    将长度 S 均匀切成 g 段，返回 [(s0,e0), (s1,e1), ...]
    """
    edges = torch.linspace(0, S, steps=g + 1).round().long().tolist()
    return [(edges[i], edges[i + 1]) for i in range(g)]


# =========================
# 3) Patch-NetVLAD 分类模型
# =========================
class PatchNetVLADClassifier(nn.Module):
    def __init__(self, num_classes, K=16, proj_dim=128, grids=(1, 2, 3), pretrained=False):
        super().__init__()
        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )  # 输出约 [B,512,7,7] (输入224)

        # 通道降维
        self.project = nn.Conv2d(512, proj_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(proj_dim)
        self.relu = nn.ReLU(inplace=True)

        # 共享 NetVLAD
        self.vlad = NetVLAD(num_clusters=K, dim=proj_dim, normalize_input=True)

        self.grids = list(grids)
        total_regions = sum(g * g for g in self.grids)
        feat_dim = total_regions * (K * proj_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        fm = self.backbone(x)                             # [B,512,H,W]
        fm = self.relu(self.bn_proj(self.project(fm)))    # [B,proj_dim,H,W]

        B, C, H, W = fm.shape
        parts = []

        # 多尺度网格区域，所有区域共享 self.vlad 权重
        for g in self.grids:
            ys = grid_bounds(H, g)
            xs = grid_bounds(W, g)
            for (y0, y1) in ys:
                for (x0, x1) in xs:
                    r = fm[:, :, y0:y1, x0:x1]
                    if r.size(-1) == 0 or r.size(-2) == 0:
                        # 极端情况下避免空区域
                        continue
                    parts.append(self.vlad(r))           # [B, K*proj_dim]

        # 若出现极端情况（某尺度为空），兜个底
        if len(parts) == 0:
            parts.append(self.vlad(fm))

        v = torch.cat(parts, dim=1)                      # [B, feat_dim]
        logits = self.head(v)
        return logits


# =========================
# 4) 训练工具
# =========================
def set_backbone_trainable(model: PatchNetVLADClassifier, flag: bool):
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
    model.train() if train else model.eval()

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
        preds = logits.argmax(dim=1)
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


# =========================
# 5) 主函数
# =========================
def main(args):
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)
    set_seed_all(args.seed)

    # ---- 数据 ----
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

    # 可选：限制样本数
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

    # ---- 模型 & 优化器 ----
    model = PatchNetVLADClassifier(
        num_classes=num_classes,
        K=args.vlad_K,
        proj_dim=args.proj_dim,
        grids=args.grids,
        pretrained=args.use_pretrained,
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

    best_f1 = 0.0

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # 可选：前若干 epoch 冻结 backbone
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
            torch.save(model.state_dict(), args.save_model)
            print(
                f"[INFO] New best F1={best_f1:.4f}, model saved to {args.save_model}",
                file=sys.stderr,
            )

    # ---- 最终评估 + 混淆矩阵 ----
    if os.path.isfile(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=device))
        print(f"[INFO] Loaded best model from {args.save_model}", file=sys.stderr)

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
        plt.title("Confusion Matrix (Patch-NetVLAD Classifier)")
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
        plt.title("Confusion Matrix (Patch-NetVLAD Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

