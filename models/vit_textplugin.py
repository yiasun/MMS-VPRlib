# 文件名：vit_textplugin.py
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# 命令行参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("ViT + BERT 文本先验 插件式多模态分类（命令行版）")
    # 数据
    p.add_argument("--data_dir", type=str, default="../all", help="图像根目录：子文件夹名即类别")
    p.add_argument("--image_size", type=int, default=32, help="图像缩放尺寸 (image_size x image_size)")
    p.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    p.add_argument("--batch_size", type=int, default=32, help="batch size")
    p.add_argument("--limit", type=int, default=0, help="仅使用前 N 张图像做实验（0 表示全量）")
    # 文本 / BERT
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="Excel 文件路径")
    p.add_argument("--text_col", type=str, default="List of Store Names", help="用于先验的文本列名")
    p.add_argument("--row_index", type=int, default=0, help="使用第几行文本（从 0 开始）")
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    # ViT 结构
    p.add_argument("--patch_size", type=int, default=4, help="Patch 大小")
    p.add_argument("--embed_dim", type=int, default=64, help="Patch embedding 维度")
    p.add_argument("--num_layers", type=int, default=6, help="Transformer Encoder 层数")
    p.add_argument("--num_heads", type=int, default=8, help="多头注意力头数")
    p.add_argument("--mlp_dim", type=int, default=128, help="Encoder 内部 FFN 维度")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    # 训练
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--step_size", type=int, default=50, help="StepLR 的 step_size")
    p.add_argument("--gamma", type=float, default=0.1, help="StepLR 的 gamma")
    # 设备 / 复现 / 输出
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="设备选择")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    p.add_argument("--save_model", type=str, default="", help="若非空，保存模型到该路径（.pth）")
    p.add_argument("--save_cm", type=str, default="", help="若非空，保存混淆矩阵图到该路径（.png）")
    return p.parse_args()


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(dev: str):
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


# =========================
# ViT 模块
# =========================
class PatchEmbedding(nn.Module):
    """
    将输入图像分割为 patch，并映射为 patch embedding
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class ViTPlugin(nn.Module):
    """
    ViT + 文本先验（BERT CLS 向量）拼接到 CLS token 输出上
    """
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_channels=1,
                 embed_dim=64,
                 num_layers=6,
                 num_heads=8,
                 mlp_dim=128,
                 num_classes=10,
                 dropout=0.1,
                 text_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 分类 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码（包含 cls）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=False,  # 我们会用 (S, B, E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头：拼接 CLS + 文本向量
        self.head = nn.Linear(embed_dim + text_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, text_vec):
        # x: (B, 1, H, W)
        B = x.size(0)
        x = self.patch_embed(x)                  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)    # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer: (S,B,E)
        x = x.transpose(0, 1)                    # (S, B, E)
        x = self.transformer(x)                  # (S, B, E)
        x = x.transpose(0, 1)                    # (B, S, E)
        x = self.norm(x)
        cls_out = x[:, 0]                        # (B, embed_dim)

        # 文本向量扩展: (text_dim,) -> (B, text_dim)
        txt = text_vec.unsqueeze(0).expand(B, -1)
        fused = torch.cat([cls_out, txt], dim=1)  # (B, embed_dim+text_dim)
        logits = self.head(fused)
        return logits


# =========================
# 主流程
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[Info] Using device: {device}")

    # ---------- 1) 读取图像 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"图像目录不存在: {args.data_dir}")

    images = []
    labels = []
    class_dirs = [d for d in os.listdir(args.data_dir)
                  if os.path.isdir(os.path.join(args.data_dir, d))]
    class_dirs.sort()
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    count = 0

    for cls in class_dirs:
        cls_path = os.path.join(args.data_dir, cls)
        for fn in os.listdir(cls_path):
            if not fn.lower().endswith(valid_ext):
                continue
            p = os.path.join(cls_path, fn)
            try:
                img = Image.open(p).convert("L")
                img = img.resize((args.image_size, args.image_size))
                arr = np.array(img, dtype=np.float32) / 255.0   # (H,W)
                images.append(arr)
                labels.append(cls)
                count += 1
                if args.limit and count >= args.limit:
                    break
            except Exception as e:
                print(f"[警告] 读取失败，跳过 {p}，错误：{e}")
        if args.limit and count >= args.limit:
            break

    if not images:
        raise RuntimeError("未成功读取到任何图像。")

    images = np.stack(images)                     # (N, H, W)
    labels = np.array(labels)
    print(f"[Info] 图像数={images.shape[0]}, 尺寸={images.shape[1:]}")

    # 标签编码
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"[Info] 类别数={num_classes}, 类别={list(le.classes_)}")

    # (N,1,H,W)
    images = np.expand_dims(images, axis=1)
    x_all = torch.tensor(images, dtype=torch.float32)       # (N,1,H,W)
    y_all = torch.tensor(labels_encoded, dtype=torch.long)  # (N,)

    # 划分训练/测试
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels_encoded
    )

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # ---------- 2) 文本先验（BERT [CLS]） ----------
    if not os.path.isfile(args.text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{args.text_data_path}")
    df_text = pd.read_excel(args.text_data_path)

    if args.text_col not in df_text.columns:
        raise KeyError(f"Excel 中未找到列 '{args.text_col}'")
    if not (0 <= args.row_index < len(df_text)):
        raise IndexError(f"--row_index 超出范围 (0 ~ {len(df_text)-1})")

    first_text = str(df_text[args.text_col].iloc[args.row_index])
    print("[Info] 文本先验内容：", first_text)

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    bert_model = BertModel.from_pretrained(args.bert_name).to(device)
    bert_model.eval()
    with torch.no_grad():
        inputs = tokenizer(first_text, return_tensors="pt",
                           truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        text_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
    text_vec = text_vec.to(device)
    text_dim = text_vec.numel()
    print(f"[Info] 文本向量维度: {text_dim}")

    # ---------- 3) 构建 ViT 模型 ----------
    model = ViTPlugin(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=1,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        text_dim=text_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )
    criterion = nn.CrossEntropyLoss()

    # ---------- 4) 训练 ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(batch_x, text_vec)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total += bs

        scheduler.step()
        avg_loss = total_loss / max(total, 1)

        if epoch == 1 or epoch == args.epochs or epoch % 20 == 0:
            # 简单测试集准确率
            model.eval()
            correct, total_eval = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    logits = model(batch_x, text_vec)
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total_eval += batch_y.size(0)
            acc = correct / max(total_eval, 1)
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | TestAcc {acc:.4f}")

    # ---------- 5) 最终评估 ----------
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x, text_vec)
            preds = logits.argmax(dim=1)
            y_true.extend(batch_y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"最终测试集 — Accuracy : {acc:.4f}")
    print(f"最终测试集 — Precision: {prec:.4f}")
    print(f"最终测试集 — Recall   : {rec:.4f}")
    print(f"最终测试集 — F1-score : {f1:.4f}")

    # ---------- 6) 混淆矩阵（可选） ----------
    if args.save_cm:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix (ViT + Text Plugin)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(num_classes), le.classes_, rotation=45)
        plt.yticks(range(num_classes), le.classes_)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(args.save_cm, dpi=150)
        plt.close()
        print(f"[Info] 混淆矩阵已保存到：{args.save_cm}")

    # ---------- 7) 保存模型（可选） ----------
    if args.save_model:
        torch.save(
            {
                "model_state": model.state_dict(),
                "label_encoder_classes": le.classes_,
                "args": vars(args)
            },
            args.save_model
        )
        print(f"[Info] 模型已保存到：{args.save_model}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
"""# 默认参数
python vit_textplugin.py

# 调一版更激进的配置并保存结果
python vit_textplugin.py \
  --data_dir "../all" \
  --image_size 32 \
  --patch_size 4 \
  --embed_dim 64 \
  --num_layers 6 \
  --num_heads 8 \
  --mlp_dim 128 \
  --dropout 0.1 \
  --batch_size 64 \
  --epochs 100 \
  --lr 0.001 \
  --step_size 50 \
  --gamma 0.1 \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --row_index 0 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --seed 42 \
  --limit 20000 \
  --save_model "vit_textplugin.pth" \
  --save_cm "cm_vit_textplugin.png"
"""