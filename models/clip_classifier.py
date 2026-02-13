# 文件名：clip_classifier.py
import os
import sys
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import clip
from tqdm import tqdm


# =========================
# 参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("CLIP 图像分类微调（命令行可调参版）")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录，子文件夹名为类别")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--batch_size", type=int, default=32,
                   help="batch size")
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader num_workers")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图片（0 表示不限制，用全量）")

    # 模型 & 训练
    p.add_argument("--backbone", type=str, default="ViT-B/32",
                   help="CLIP 主干名称，例如 ViT-B/32, RN50 等")
    p.add_argument("--epochs", type=int, default=10,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="学习率（分类头 & 解冻层共用）")
    p.add_argument("--unfreeze_blocks", type=int, default=1,
                   help="解冻 CLIP visual transformer 最后多少个 block（0=仅训练分类头）")

    # 设备 & 复现 & 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="设备选择")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    p.add_argument("--save_model", type=str, default="",
                   help="若非空，保存模型到该路径（.pth）")

    return p.parse_args()


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(arg: str):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# =========================
# 数据集
# =========================
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# =========================
# 模型定义
# =========================
class CLIPClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x 已经按 CLIP 预处理
        img_emb = self.backbone.encode_image(x)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        logits = self.classifier(img_emb)
        return logits


# =========================
# 主流程
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    # ---------- 1) 构建类别 & 文件列表 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    classes = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])
    if not classes:
        raise RuntimeError("未找到任何类别子目录，请检查 data_dir 设置。")

    le = LabelEncoder()
    le.fit(classes)
    num_classes = len(classes)

    file_paths = []
    labels = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    count = 0

    for cls in classes:
        cls_dir = os.path.join(args.data_dir, cls)
        cls_idx = int(le.transform([cls])[0])
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(valid_ext):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
                count += 1
                if args.limit and count >= args.limit:
                    break
        if args.limit and count >= args.limit:
            break

    if not file_paths:
        raise RuntimeError("未找到任何图像文件，请检查数据集路径和格式。")

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    print(f"[INFO] Loaded {len(file_paths)} images from {num_classes} classes",
          file=sys.stderr)

    # ---------- 2) 划分训练 / 测试 ----------
    idxs = np.arange(len(file_paths))
    train_idx, test_idx = train_test_split(
        idxs,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels
    )

    # ---------- 3) 定义增强与预处理 ----------
    # 使用 CLIP 官方均值方差
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    train_ds = ImageDataset(file_paths[train_idx], labels[train_idx], transform=train_transform)
    test_ds = ImageDataset(file_paths[test_idx],  labels[test_idx],  transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # ---------- 4) 加载 CLIP ----------
    print(f"[INFO] Loading CLIP backbone: {args.backbone} ...", file=sys.stderr)
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model = clip_model.float()
    print("[INFO] CLIP loaded.", file=sys.stderr)

    # 冻结所有参数
    for p in clip_model.parameters():
        p.requires_grad = False

    # 解冻最后若干个 visual transformer block（仅在 ViT 结构上有用）
    if args.unfreeze_blocks > 0 and hasattr(clip_model.visual, "transformer"):
        resblocks = clip_model.visual.transformer.resblocks
        n = min(args.unfreeze_blocks, len(resblocks))
        for block in resblocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[INFO] Unfreezing last {n} transformer blocks.", file=sys.stderr)
    else:
        print("[INFO] Only training classifier head (CLIP frozen).", file=sys.stderr)

    # ---------- 5) 构建分类模型 ----------
    embed_dim = clip_model.visual.output_dim
    model = CLIPClassifier(clip_model, embed_dim, num_classes).to(device)

    # 参数组：分类头 + 所有 requires_grad=True 的 CLIP 参数
    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    params = [
        {"params": model.classifier.parameters(), "lr": args.lr},
    ]
    if backbone_trainable:
        params.append({"params": backbone_trainable, "lr": args.lr})

    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ---------- 6) 训练 & 每轮验证 ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stderr)
        for imgs, labs in loop:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            loop.set_postfix(loss=total_loss / max(total_samples, 1))

        avg_loss = total_loss / max(total_samples, 1)

        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                y_true.extend(labs.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(
            f"[INFO] Epoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Val Prec: {prec:.4f} | "
            f"Val Rec: {rec:.4f} | "
            f"Val F1: {f1:.4f}",
            file=sys.stderr
        )

    # ---------- 7) 最终测试集评估 ----------
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            y_true.extend(labs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    final_acc = accuracy_score(y_true, y_pred)
    final_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    final_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    final_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(
        f"[RESULT] Final Test Acc: {final_acc:.4f} | "
        f"Prec: {final_prec:.4f} | "
        f"Rec: {final_rec:.4f} | "
        f"F1: {final_f1:.4f}"
    )

    # ---------- 8) 保存模型（可选） ----------
    if args.save_model:
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": le.classes_,
                "args": vars(args),
            },
            args.save_model
        )
        print(f"[INFO] Model saved to {args.save_model}", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    main(args)
