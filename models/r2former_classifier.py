#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R2Former-style Classifier (CLI Version)

Backbone:
  - ResNet18 至 layer4, 输出特征图

Tokenizer:
  - 1x1 Conv: 512 -> embed_dim
  - 展平为 N = H*W 个 patch token

Positional Encoding:
  - 动态 2D sin-cos 位置编码 (依赖特征图 H,W)

Transformer Encoder:
  - nn.TransformerEncoder (norm_first, GELU)

R2 Aggregation:
  - 使用 [CLS] 与所有 patch token 的相似度作为权重进行加权聚合
  - 最终特征 = sigmoid(alpha)*CLS + (1-sigmoid(alpha))*patch_agg

Head:
  - BN + Dropout + Linear -> num_classes

提供指标:
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


# =========================
# 参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("R2Former-style Classifier")

    # 数据相关
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录（子文件夹为类别）")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入图像尺寸，默认 224")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例，默认 0.2")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 的 num_workers，Windows 建议 0")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图片做实验，0 表示不限制")

    # 训练相关
    p.add_argument("--batch_size", type=int, default=32,
                   help="batch size，默认 32")
    p.add_argument("--epochs", type=int, default=10,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="权重衰减")
    p.add_argument("--use_pretrained", action="store_true",
                   help="使用 ImageNet 预训练 ResNet18")
    p.add_argument("--use_amp", action="store_true",
                   help="开启混合精度训练（需 CUDA）")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前 K 个 epoch 冻结 backbone，仅训练 Transformer+head")

    # Transformer / 聚合超参
    p.add_argument("--embed_dim", type=int, default=256,
                   help="token embedding 维度 (由 512 降维)")
    p.add_argument("--depth", type=int, default=4,
                   help="TransformerEncoder 层数")
    p.add_argument("--num_heads", type=int, default=8,
                   help="多头注意力头数，需整除 embed_dim")
    p.add_argument("--mlp_ratio", type=float, default=4.0,
                   help="FFN 隐层维度 = embed_dim * mlp_ratio")
    p.add_argument("--drop_rate", type=float, default=0.1,
                   help="Dropout 概率")
    p.add_argument("--alpha_init", type=float, default=0.5,
                   help="R2 融合中 alpha 初始值 (0~1 附近)")

    # 设备 / 随机数 / 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="运行设备：auto / cpu / cuda")
    p.add_argument("--seed", type=int, default=42,
                   help="全局随机种子")
    p.add_argument("--split_seed", type=int, default=42,
                   help="train/test 划分随机种子")
    p.add_argument("--save_model", type=str, default="r2former_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="若非空，则保存混淆矩阵为该路径；空则直接 plt.show()")

    return p.parse_args()


# =========================
# 工具函数
# =========================
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


# =========================
# 2D 正弦-余弦位置编码
# =========================
def get_2d_sincos_pos_embed(embed_dim: int, H: int, W: int, device=None):
    """
    返回 [1, H*W, D] 的 2D sin-cos 位置编码
    要求 embed_dim 是 4 的倍数
    """
    assert embed_dim % 4 == 0, "embed_dim 必须是 4 的倍数以构造 2D sin-cos 编码"
    if device is None:
        device = "cpu"

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    dim_each = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(dim_each, device=device).float() / dim_each))  # [d]

    yy = yy.reshape(-1, 1) * omega.view(1, -1)  # [H*W, d]
    xx = xx.reshape(-1, 1) * omega.view(1, -1)  # [H*W, d]

    pos = torch.cat(
        [torch.sin(yy), torch.cos(yy), torch.sin(xx), torch.cos(xx)], dim=1
    )  # [H*W, D]

    return pos.unsqueeze(0)  # [1, H*W, D]


# =========================
# 模型定义
# =========================
class R2FormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        pretrained=False,
        alpha_init=0.5,
    ):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError("embed_dim 必须是 4 的倍数，用于 2D 位置编码")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        # ResNet18 backbone
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
        )  # 输出 [B,512,Hf,Wf]

        # 降维到 embed_dim
        self.reduce = nn.Conv2d(512, embed_dim, kernel_size=1, bias=False)
        self.bn_red = nn.BatchNorm2d(embed_dim)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # R2 聚合的 alpha（经过 sigmoid 后落在 (0,1)）
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # 分类头
        self.bn_vec = nn.BatchNorm1d(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        # CNN 特征
        x = self.backbone(x)                       # [B,512,Hf,Wf]
        x = F.gelu(self.bn_red(self.reduce(x)))    # [B,D,Hf,Wf]
        B, D, H, W = x.shape
        N = H * W

        # patch tokens
        tokens = x.flatten(2).transpose(1, 2)      # [B,N,D]

        # 2D 位置编码
        pos = get_2d_sincos_pos_embed(D, H, W, device=x.device)  # [1,N,D]
        tokens = tokens + pos

        # [CLS] + tokens
        cls = self.cls_token.expand(B, 1, D)       # [B,1,D]
        seq = torch.cat([cls, tokens], dim=1)      # [B,1+N,D]

        # Transformer Encoder
        y = self.encoder(seq)                      # [B,1+N,D]
        cls_out = y[:, 0, :]                       # [B,D]
        patch_out = y[:, 1:, :]                    # [B,N,D]

        # R2：基于和 CLS 的相似度进行 patch 聚合
        scores = torch.einsum("bd,bnd->bn", cls_out, patch_out) / (D ** 0.5)  # [B,N]
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)                     # [B,N,1]
        patch_agg = torch.sum(patch_out * attn, dim=1)                        # [B,D]

        a = torch.sigmoid(self.alpha)
        vec = a * cls_out + (1.0 - a) * patch_agg                             # [B,D]

        vec = self.bn_vec(vec)
        logits = self.head(vec)                                               # [B,num_classes]
        return logits


# =========================
# 训练 / 验证
# =========================
def set_backbone_trainable(model: R2FormerClassifier, flag: bool):
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


# =========================
# 主流程
# =========================
def main(args):
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    set_seed_all(args.seed)

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
        raise RuntimeError("类别数 <=1，请检查数据集目录结构（子文件夹=类别）")

    indices = np.arange(len(full_set))
    labels = np.array(full_set.targets)

    # 可选限制样本总数（方便快速调试）
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

    # DataLoader（给 train 的 shuffle 单独 rng，便于方案B随机性）
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
    model = R2FormerClassifier(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop_rate,
        pretrained=args.use_pretrained,
        alpha_init=args.alpha_init,
    ).to(device)

    print(model, file=sys.stderr)

    # ---------- 优化器 / 调度器 ----------
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

    # ---------- 训练 ----------
    for epoch in range(1, args.epochs + 1):
        # 可选：冻结 backbone 若干轮
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
        plt.title("Confusion Matrix (R2Former-style Classifier)")
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
        plt.title("Confusion Matrix (R2Former-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
