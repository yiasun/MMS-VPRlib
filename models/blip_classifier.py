# 文件名：blip_classifier.py
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import BlipProcessor, BlipModel
from tqdm import tqdm


# =========================
# 命令行参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("BLIP 视觉编码 + 线性分类头 图像分类（命令行版）")

    # 数据
    p.add_argument("--data_dir", type=str, default="../all",
                   help="图像根目录，子文件夹名为类别")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--batch_size", type=int, default=16,
                   help="batch size")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers（Win 上建议 0）")
    p.add_argument("--limit", type=int, default=0,
                   help="仅使用前 N 张图片做实验（0 表示全量）")

    # 图像增强
    p.add_argument("--image_size", type=int, default=384,
                   help="输入 BLIP 的图像尺寸（默认 384）")
    p.add_argument("--aug_scale_min", type=float, default=0.8,
                   help="RandomResizedCrop 最小 scale")

    # 模型 & 训练
    p.add_argument("--blip_name", type=str,
                   default="Salesforce/blip-image-captioning-base",
                   help="BLIP 模型名")
    p.add_argument("--epochs", type=int, default=8,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="学习率（仅分类头）")

    # 设备 & 复现 & 输出
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="设备选择")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    p.add_argument("--save_model", type=str, default="best_blip_classifier.pth",
                   help="最佳模型保存路径")
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
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# =========================
# 模型定义
# =========================
class BlipClassifier(nn.Module):
    """
    使用 BLIP 的 vision_model 提取图像特征 + 线性分类头
    （只训练分类头，BLIP 冻结）
    """
    def __init__(self, blip_model, feature_dim, num_labels):
        super().__init__()
        self.blip = blip_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_labels),
        )

    def forward(self, x):
        # 直接走 vision_model
        outputs = self.blip.vision_model(pixel_values=x)
        img_embeds = outputs.pooler_output  # (B, feature_dim)
        logits = self.classifier(img_embeds)
        return logits


# =========================
# 主流程
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    # ---------- 1) 读取数据 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    classes = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )
    if not classes:
        raise RuntimeError("未找到任何类别子目录，请检查 data_dir 设置。")

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    num_classes = len(classes)

    file_paths, labels = [], []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    count = 0

    for cls in classes:
        cls_idx = class_to_idx[cls]
        cls_dir = os.path.join(args.data_dir, cls)
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
        raise RuntimeError("未发现任何图像文件，请检查路径/格式。")

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
        stratify=labels,
    )

    # ---------- 3) 图像增强 ----------
    # 这里使用 ImageNet 风格的均值方差，与大部分 ViT/ResNet 一致
    # BLIP 官方推荐用 processor；你也可以改成用 BlipProcessor 预处理。
    image_size = args.image_size
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(args.aug_scale_min, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = ImageDataset(file_paths[train_idx], labels[train_idx], transform=train_transform)
    test_ds = ImageDataset(file_paths[test_idx], labels[test_idx], transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------- 4) 加载 BLIP ----------
    print(f"[INFO] Loading BLIP model: {args.blip_name}", file=sys.stderr)
    # processor 保留着，方便你之后切到文本+图像模式用
    _ = BlipProcessor.from_pretrained(args.blip_name)
    blip = BlipModel.from_pretrained(args.blip_name)
    blip.to(device)

    # 冻结 BLIP 全部参数，只训分类头
    for p in blip.parameters():
        p.requires_grad = False
    blip.eval()

    vision_dim = blip.vision_model.config.hidden_size
    print(f"[INFO] BLIP vision hidden size = {vision_dim}", file=sys.stderr)

    # ---------- 5) 构建分类模型 ----------
    model = BlipClassifier(blip, vision_dim, num_classes).to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ---------- 6) 训练 ----------
    best_f1 = 0.0
    best_path = args.save_model if args.save_model else "best_blip_classifier.pth"

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

        # ---------- 验证 ----------
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True)
                preds = model(imgs).argmax(dim=1)
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

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best F1={best_f1:.4f}, saved to {best_path}",
                  file=sys.stderr)

    # ---------- 7) 最终测试评估（加载最佳权重） ----------
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}", file=sys.stderr)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            preds = model(imgs).argmax(dim=1)
            y_true.extend(labs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    final_acc = accuracy_score(y_true, y_pred)
    final_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    final_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    final_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(
        f"[RESULT] Final Test | "
        f"Acc: {final_acc:.4f} | "
        f"Prec: {final_prec:.4f} | "
        f"Rec: {final_rec:.4f} | "
        f"F1: {final_f1:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main()

"""# 默认参数
python blip_classifier.py

# 控制采样数量 + 自定义学习率 + 自定义保存路径
python blip_classifier.py \
  --data_dir "../all" \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-4 \
  --limit 20000 \
  --save_model "blip_best.pth"
"""