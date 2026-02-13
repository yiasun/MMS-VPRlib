#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CosPlace-style Classifier (CLI Version)

Backbone:
  - ResNet18 到 layer4

Pooling:
  - GeM (learnable p)

Head:
  - CosFace / AM-Softmax 余弦分类头 (scale s, margin m)

Loss:
  - CrossEntropy on CosFace logits (训练时带 margin, 测试时不带 margin)

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


# ==============================
# 0) 参数
# ==============================
def parse_args():
    p = argparse.ArgumentParser("CosPlace-style Classifier")

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
                   help="仅使用前 N 张图片做实验，0 表示不限制")

    # 训练
    p.add_argument("--batch_size", type[int], default=32,
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
                   help="前 K 个 epoch 冻结骨干网络，只训练头部")

    # CosFace / AM-Softmax 超参
    p.add_argument("--cos_s", type=float, default=30.0,
                   help="CosFace/AM-Softmax 缩放 s")
    p.add_argument("--cos_m", type=float, default=0.35,
                   help="CosFace/AM-Softmax 边距 m")

    # 设备 / 随机 / 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="运行设备")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（用于数据划分等）")
    p.add_argument("--split_seed", type=int, default=42,
                   help="train/test 划分种子")
    p.add_argument("--save_model", type=str, default="cosplace_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="混淆矩阵保存路径（png）；为空则弹窗显示")

    return p.parse_args()


def select_device(dev_arg: str):
    if dev_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_arg)


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


# ==============================
# 1) GeM 池化
# ==============================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learnable)
        self.eps = eps

    def forward(self, x):  # [B,C,H,W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / self.p)
        return x.view(x.size(0), -1)  # [B,C]


# ==============================
# 2) 余弦分类头
# ==============================
class CosMarginProduct(nn.Module):
    """
    训练:
        logits = s * (cos_theta - m * one_hot(y))
    推理/验证:
        labels=None -> logits = s * cos_theta
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        # x: [B,D]
        x_n = F.normalize(x, p=2, dim=1)
        W_n = F.normalize(self.weight, p=2, dim=1)  # [C,D]
        cos_theta = torch.matmul(x_n, W_n.t())      # [B,C]

        if labels is None:
            return self.s * cos_theta

        one_hot = F.one_hot(labels, num_classes=self.out_features).float()
        logits = self.s * (cos_theta - one_hot * self.m)
        return logits


# ==============================
# 3) 模型
# ==============================
class CosPlaceClassifier(nn.Module):
    def __init__(self, num_classes, s=30.0, m=0.35, pretrained=False):
        super().__init__()
        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1, self.layer2 = res.layer1, res.layer2
        self.layer3, self.layer4 = res.layer3, res.layer4

        self.gem = GeM(p=3.0, learnable=True)
        self.feat_dim = 512
        self.bn_feat = nn.BatchNorm1d(self.feat_dim)

        self.head = CosMarginProduct(
            in_features=self.feat_dim,
            out_features=num_classes,
            s=s,
            m=m,
        )

    def forward_feats(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # [B,512,H,W]
        x = self.gem(x)         # [B,512]
        x = self.bn_feat(x)
        return x

    def forward(self, x, labels=None):
        feats = self.forward_feats(x)
        logits = self.head(feats, labels)
        return logits


# ==============================
# 4) 训练与评测
# ==============================
def set_backbone_trainable(model: CosPlaceClassifier, flag: bool):
    for m in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters():
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
                    # 训练：带 margin；验证/测试：不带 margin
                    logits = model(imgs, labels if train else None)
                    loss = criterion(logits, labels)
            else:
                logits = model(imgs, labels if train else None)
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


def main(args):
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)
    set_seed_all(args.seed)

    # ===== 数据 =====
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
        raise RuntimeError("类别数 <= 1，请检查数据集结构（子文件夹=类别）。")

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

    # ===== 模型 & 优化器 =====
    model = CosPlaceClassifier(
        num_classes=num_classes,
        s=args.cos_s,
        m=args.cos_m,
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

    # ===== 训练循环 =====
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

    # ===== 最终评估 + 混淆矩阵 =====
    if os.path.isfile(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=device))
        print(f"[INFO] Loaded best model from {args.save_model}", file=sys.stderr)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs, labels=None)  # 测试不加 margin
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    print(f"[INFO] Best macro-F1: {best_f1:.4f}")

    if args.save_cm:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (CosPlace-style Classifier)")
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
        plt.title("Confusion Matrix (CosPlace-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

