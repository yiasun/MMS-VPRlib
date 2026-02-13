#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SALAD-style Classifier
分层选择性聚合 + 蒸馏
命令行可调参与固定划分版本
"""

import os
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
# 命令行参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("SALAD-style 分层选择性聚合 + 蒸馏 分类器")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录，子文件夹名为类别 (ImageFolder 格式)")
    p.add_argument("--image_size", type=int, default=224,
                   help="输入图像尺寸 (image_size x image_size)")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers (Win 建议 0)")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图片做实验，0 表示不限制")

    # 训练参数
    p.add_argument("--batch_size", type=int, default=32,
                   help="batch size")
    p.add_argument("--epochs", type=int, default=10,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="初始学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="权重衰减")
    p.add_argument("--use_pretrained", action="store_true",
                   help="是否使用 ImageNet 预训练 ResNet18")
    p.add_argument("--use_amp", action="store_true",
                   help="是否启用混合精度训练 (cuda 有效)")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0,
                   help="前若干个 epoch 冻结 backbone，只训练聚合和分类头 (0 表示不冻结)")

    # SALAD / 蒸馏超参
    p.add_argument("--proj_dim", type=int, default=192,
                   help="各层降维后的通道数")
    p.add_argument("--topk_ratio", type=float, default=0.25,
                   help="每层选择性池化保留的 Top-K 比例")
    p.add_argument("--min_topk", type=int, default=4,
                   help="每层至少保留的 Top-K 数")
    p.add_argument("--score_hidden", type=int, default=96,
                   help="显著性评分支路的中间通道数")
    p.add_argument("--drop_rate", type=float, default=0.1,
                   help="选择性聚合/全连接的 dropout")
    p.add_argument("--use_distill", action="store_true",
                   help="是否启用分层蒸馏损失")
    p.add_argument("--lambda_distill", type=float, default=5e-4,
                   help="蒸馏损失权重")
    p.add_argument("--clip_grad_norm", type=float, default=0.0,
                   help="梯度裁剪阈值，<=0 表示不裁剪")

    # 随机性 / 设备 / 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="设备选择")
    p.add_argument("--seed", type=int, default=-1,
                   help="全局随机种子；<0 表示每次随机 (方案B)")
    p.add_argument("--split_seed", type=int, default=42,
                   help="仅用于 train/test 划分的随机种子")
    p.add_argument("--save_model", type=str, default="salad_best.pt",
                   help="最佳模型保存路径")
    p.add_argument("--save_cm", type=str, default="",
                   help="若非空，保存混淆矩阵图到该路径")
    return p.parse_args()


# =========================
# 工具函数
# =========================
def select_device(arg: str):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def set_seed_dynamic_or_fixed(seed_arg: int):
    """方案B：seed<0 时每次运行不同；>=0 时固定（方便复现）"""
    if seed_arg is None or seed_arg < 0:
        # 动态随机种子
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

    print(f"[INFO] Seeds -> numpy:{np_seed} torch:{th_seed}")
    return np_seed, th_seed


# =========================
# 模块定义
# =========================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learnable)
        self.eps = eps

    def forward(self, x):  # [B,C,H,W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / self.p)
        return x.view(x.size(0), -1)


class SelectivePool(nn.Module):
    """
    显著性评分 -> Top-K 位置 -> softmax 加权聚合
    输入: x ∈ [B,C,H,W] -> 输出: v ∈ [B,C], score_map ∈ [B,1,H,W]
    """
    def __init__(self, in_channels, hidden=96, keep_ratio=0.25, min_topk=4, drop=0.0):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.min_topk = min_topk
        self.scorer = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )
        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        score = self.scorer(x).view(B, N)   # [B,N]

        K = max(self.min_topk, int(N * self.keep_ratio))
        K = min(K, N)
        top_vals, top_idx = torch.topk(score, k=K, dim=1, largest=True, sorted=False)  # [B,K]

        x_flat = x.view(B, C, N)
        idx_exp = top_idx.unsqueeze(1).expand(-1, C, -1)  # [B,C,K]
        sel = x_flat.gather(2, idx_exp)                   # [B,C,K]
        w = torch.softmax(top_vals, dim=1).unsqueeze(1)   # [B,1,K]
        v = torch.sum(sel * w, dim=2)                     # [B,C]
        v = self.drop(v)
        return v, score.view(B, 1, H, W)


class SALADClassifier(nn.Module):
    """
    ResNet18 backbone + 多层选择性聚合 + GeM teacher 蒸馏
    """
    def __init__(
        self,
        num_classes,
        proj_dim=192,
        hidden=96,
        keep_ratio=0.25,
        min_topk=4,
        drop=0.1,
        pretrained=False,
    ):
        super().__init__()
        res = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # backbone
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        self.relu = nn.ReLU(inplace=True)

        # 各层降维到统一 proj_dim
        self.proj2 = nn.Conv2d(128, proj_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(256, proj_dim, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(512, proj_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(proj_dim)
        self.bn3 = nn.BatchNorm2d(proj_dim)
        self.bn4 = nn.BatchNorm2d(proj_dim)

        # 选择性池化
        self.sel2 = SelectivePool(
            proj_dim, hidden=hidden,
            keep_ratio=keep_ratio, min_topk=min_topk, drop=drop
        )
        self.sel3 = SelectivePool(
            proj_dim, hidden=hidden,
            keep_ratio=keep_ratio, min_topk=min_topk, drop=drop
        )
        self.sel4 = SelectivePool(
            proj_dim, hidden=hidden,
            keep_ratio=keep_ratio, min_topk=min_topk, drop=drop
        )

        # 层级门控
        self.g2 = nn.Parameter(torch.tensor(1.0))
        self.g3 = nn.Parameter(torch.tensor(1.0))
        self.g4 = nn.Parameter(torch.tensor(1.0))

        # 融合后的维度
        fused_dim = proj_dim * 3
        self.bn_vec = nn.BatchNorm1d(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(fused_dim, num_classes),
        )

        # Teacher: 对 layer4 用 GeM，输出 512 维
        self.teacher_gem = GeM(p=3.0, learnable=True)

        # 蒸馏用投影：将学生向量映射到 512 维
        self.distill_proj = nn.Linear(fused_dim, 512, bias=False)
        nn.init.xavier_uniform_(self.distill_proj.weight)

    def forward_feats(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)          # (B,64,...)
        x2 = self.layer2(x1)         # (B,128,...)
        x3 = self.layer3(x2)         # (B,256,...)
        x4 = self.layer4(x3)         # (B,512,...)

        p2 = self.relu(self.bn2(self.proj2(x2)))
        p3 = self.relu(self.bn3(self.proj3(x3)))
        p4 = self.relu(self.bn4(self.proj4(x4)))

        v2, _ = self.sel2(p2)
        v3, _ = self.sel3(p3)
        v4, _ = self.sel4(p4)

        v = torch.cat(
            [self.g2 * v2, self.g3 * v3, self.g4 * v4],
            dim=1
        )  # (B, 3*proj_dim)
        return v, x4

    def forward(self, x):
        v, x4 = self.forward_feats(x)
        v_bn = self.bn_vec(v)
        logits = self.head(v_bn)

        # teacher embedding: GeM(layer4) + L2 norm
        t = F.normalize(self.teacher_gem(x4), p=2, dim=1)  # (B,512)
        return logits, v_bn, t  # 注意返回的是 BN 后的 v 用于蒸馏


def distill_loss(model: SALADClassifier, student_vec, teacher_vec):
    """
    1 - cos( W * s, t )
    s: 学生融合向量 (已 BN)，t: teacher_vec (已归一化 512维)
    """
    s = F.normalize(student_vec, p=2, dim=1)
    s_mapped = F.normalize(model.distill_proj(s), p=2, dim=1)
    cos = torch.sum(s_mapped * teacher_vec, dim=1)  # [B]
    return torch.mean(1.0 - cos)


# =========================
# 训练 / 验证
# =========================
def set_backbone_trainable(model: SALADClassifier, flag: bool):
    for m in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters():
            p.requires_grad = flag


def run_epoch(
    model,
    loader,
    device,
    criterion,
    use_distill: bool,
    lambda_distill: float,
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
                    logits, v, t = model(imgs)
                    ce = criterion(logits, labels)
                    loss = ce + (lambda_distill * distill_loss(model, v, t)
                                 if use_distill else 0.0)
            else:
                logits, v, t = model(imgs)
                ce = criterion(logits, labels)
                loss = ce + (lambda_distill * distill_loss(model, v, t)
                             if use_distill else 0.0)

            if train and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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
    print(f"[INFO] Using device: {device}")

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
        raise RuntimeError("类别数 <=1，检查数据集目录结构（需要子文件夹按类别命名）")

    indices = np.arange(len(full_set))
    labels = np.array(full_set.targets)

    # 可选限制样本数量
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

    # DataLoader 独立随机源（仅影响 shuffle）
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

    print(f"[INFO] Classes ({num_classes}): {full_set.classes}")
    print(f"[INFO] Train: {len(train_set)} | Test: {len(test_set)}")

    # ---------- 模型 ----------
    model = SALADClassifier(
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        hidden=args.score_hidden,
        keep_ratio=args.topk_ratio,
        min_topk=args.min_topk,
        drop=args.drop_rate,
        pretrained=args.use_pretrained,
    ).to(device)

    print(model)

    # ---------- 优化器 & 调度器 ----------
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

    # ---------- 训练循环 ----------
    best_f1 = 0.0
    best_path = args.save_model

    for epoch in range(1, args.epochs + 1):
        # 冻结/解冻 backbone
        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        tr = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            use_distill=args.use_distill,
            lambda_distill=args.lambda_distill,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad_norm=args.clip_grad_norm,
            train=True,
        )
        scheduler.step()
        te = run_epoch(
            model,
            test_loader,
            device,
            criterion,
            use_distill=args.use_distill,
            lambda_distill=args.lambda_distill,
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
            f"Train {tr_seen/max(tr_dt,1e-6):.1f} img/s, "
            f"Test {te_seen/max(te_dt,1e-6):.1f} img/s"
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), best_path)
            print(
                f"[INFO] New best F1={best_f1:.4f}, model saved to {best_path}"
            )

    # ---------- 最终评估 + 混淆矩阵 ----------
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits, _, _ = model(imgs)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    print(f"[INFO] Best macro-F1: {best_f1:.4f}")

    if args.save_cm:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (SALAD-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(args.save_cm, dpi=150)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {args.save_cm}")
    else:
        # 直接展示（在非交互/终端环境你可以关掉这段）
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (SALAD-style Classifier)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""# 默认配置跑一遍
python salad_classifier.py

# 用预训练、开启蒸馏、冻结 backbone 2 个 epoch、保存混淆矩阵
python salad_classifier.py \
  --data_dir "../all" \
  --use_pretrained \
  --use_distill \
  --freeze_backbone_epochs 2 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-3 \
  --lambda_distill 5e-4 \
  --clip_grad_norm 5.0 \
  --save_model "salad_best.pt" \
  --save_cm "salad_cm.png"
"""