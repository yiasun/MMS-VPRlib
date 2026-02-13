#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFRS-style Classifier

Backbone:
  - ResNet18 conv 特征 (到 layer4)

SFRS Block (简化版思想):
  - 对特征图生成空间注意力 A ∈ [0,1]
  - 去均值后用于重标定各位置响应，突出判别区域
  - 残差式融合保证稳定

Head:
  - GeM 池化 (learnable p)
  - MLP 分类头

Metrics:
  - Accuracy, macro Precision / Recall / F1, Confusion Matrix
"""

import os
import sys
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


# ======================
# 0) 配置 / CLI
# ======================
import argparse

def parse_args():
    p = argparse.ArgumentParser("SFRS-style Classifier")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录（子目录为类别）")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入尺寸，默认224")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers，Windows 建议0")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前N张图片调试，0为不限制")

    # 训练
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_pretrained", action="store_true",
                   help="使用ImageNet预训练ResNet18")
    p.add_argument("--use_amp", action="store_true",
                   help="启用AMP混合精度（需CUDA）")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前K个epoch冻结backbone，仅训练SFRS+head")

    # SFRS超参
    p.add_argument("--sfrs_reduction", type=int, default=16,
                   help="SFRS Block中的通道压缩比")

    # 通用
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42,
                   help="train/test划分随机种子")
    p.add_argument("--save_model", type=str, default="sfrs_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="混淆矩阵保存路径，为空则直接show")

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


# ======================
# 1) GeM 池化
# ======================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learnable)
        self.eps = eps

    def forward(self, x):  # [B,C,H,W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.view(x.size(0), -1)  # [B,C]


# ======================
# 2) SFRS-style Block
# ======================
class SFRSBlock(nn.Module):
    """
    空间注意力 + 稳定残差重标定：

    A = sigmoid(Conv(Conv(x)))
    A_hat = A - mean(A)          # 去全局均值，突出差异
    x' = x * (1 + gamma * A_hat)
    y  = x + beta * x'
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # [B,C,H,W]
        A = self.spatial(x)                                    # [B,1,H,W] in [0,1]
        A_hat = A - A.mean(dim=(2, 3), keepdim=True)           # 中心化
        x_prime = x * (1.0 + self.gamma * A_hat)               # 重标定
        y = x + self.beta * x_prime                            # 稳定残差
        return y


# ======================
# 3) 模型
# ======================
class SFRSClassifier(nn.Module):
    def __init__(self, num_classes: int,
                 sfrs_reduction: int = 16,
                 pretrained: bool = False):
        super().__init__()
        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        self.sfrs  = SFRSBlock(in_channels=512, reduction=sfrs_reduction)
        self.pool  = GeM(p=3.0, learnable=True)
        self.head  = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        # [B,512,H,W]
        x = self.sfrs(x)
        v = self.pool(x)          # [B,512]
        logits = self.head(v)
        return logits


# ======================
# 4) 训练工具
# ======================
def set_backbone_trainable(model: SFRSClassifier, flag: bool):
    for m in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters():
            p.requires_grad = flag


def run_epoch(
    model: SFRSClassifier,
    loader: DataLoader,
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
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1, dt, seen


# ======================
# 5) 主流程
# ======================
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
        raise RuntimeError("类别数 <= 1，请检查数据集结构。")

    indices = np.arange(len(full_set))
    labels = np.array(full_set.targets)

    # 可选：限制样本数调试
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
    test_set  = Subset(full_set, test_idx)

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
    model = SFRSClassifier(
        num_classes=num_classes,
        sfrs_reduction=args.sfrs_reduction,
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
        # 可选：冻结backbone若干epoch
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
        plt.title("Confusion Matrix (SFRS-style Classifier)")
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
        plt.title("Confusion Matrix (SFRS-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""python sfrs_classifier.py \
  --use_pretrained \
  --use_amp \
  --epochs 20 \
  --freeze_backbone_epochs 2 \
  --save_model sfrs_best.pt \
  --save_cm sfrs_cm.png
"""