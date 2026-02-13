# 文件名：resnet_bert_hetero_gated.py
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
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt


# =========================
# 参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("ResNet + BERT + HeteroGNN Gating 多模态融合分类（命令行版）")

    # 数据路径
    p.add_argument("--data_dir", type=str, default="../all", help="图像根目录，子文件夹名为类别")
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="文本 Excel 路径")

    # 图像 & Loader
    p.add_argument("--image_size", type=int, default=32, help="缩放为 (image_size, image_size)")
    p.add_argument("--batch_size", type=int, default=32, help="batch size")
    p.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    p.add_argument("--limit", type=int, default=0, help="仅使用前 N 张图片（0 为不限制）")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    # 文本 BERT
    p.add_argument("--text_cols", type=str, default="Type,List of Store Names",
                   help="用于拼接文本先验的列名，逗号分隔")
    p.add_argument("--row_index", type=int, default=0, help="取第几行作为文本先验（0 开始）")
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    p.add_argument("--unfreeze_last_n", type=int, default=2,
                   help="解冻 BERT 最后 N 层（0=全冻结，仅用预计算向量）")

    # HeteroGNN Encoder（示例）
    p.add_argument("--hetero_in_dim", type=int, default=16, help="HeteroGNN 输入维度")
    p.add_argument("--hetero_hidden_dim", type=int, default=32, help="HeteroGNN 隐层维度")
    p.add_argument("--hetero_out_dim", type=int, default=768, help="HeteroGNN 输出维度（需与 text_dim 对齐比较自然）")
    p.add_argument("--hetero_type1_nodes", type=int, default=10, help="示例图 type1 节点数")
    p.add_argument("--hetero_type2_nodes", type=int, default=15, help="示例图 type2 节点数")

    # ResNet + 训练
    p.add_argument("--epochs", type=int, default=20, help="训练轮数")
    p.add_argument("--lr_img", type=float, default=1e-3, help="ResNet 分支学习率")
    p.add_argument("--lr_bert", type=float, default=5e-5, help="BERT 微调学习率")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")

    # 设备 & 复现 & 输出
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="设备选择")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
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
# 数据集
# =========================
class ImageDataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs  # (N,H,W)
        self.lbls = lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = np.expand_dims(self.imgs[idx], 0)  # (1,H,W)
        y = self.lbls[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =========================
# HeteroGNN Encoder (示例)
# =========================
class HeteroGNNEncoder(nn.Module):
    """
    简化版异质图编码器：
    - 输入: dict{name: 节点特征张量 [N_i, in_feats]}
    - 每类节点通过两层 MLP，最后所有节点 embedding 拼接后做均值，得到全局向量。
    """
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats)
        self.fc2 = nn.Linear(hid_feats, out_feats)
        self.device = device

    def forward(self, graph_dict):
        encs = []
        for feats in graph_dict.values():
            feats = feats.to(self.device)
            h = F.relu(self.fc1(feats))
            h = self.fc2(h)
            encs.append(h)
        all_h = torch.cat(encs, dim=0)  # (sum_nodes, out_feats)
        return all_h.mean(dim=0)        # (out_feats,)


# =========================
# ResNet 主干 + Gating 融合
# =========================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.down = None
        if stride != 1 or in_planes != planes:
            self.down = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNetFusionGated(nn.Module):
    def __init__(self, block, layers, num_classes, t_dim=768, g_dim=768, in_ch=1, device=None):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        def make_layer(planes, blocks, stride):
            layers_list = [block(self.in_planes, planes, stride)]
            self.in_planes = planes * block.expansion
            for _ in range(1, blocks):
                layers_list.append(block(self.in_planes, planes))
            return nn.Sequential(*layers_list)

        self.layer1 = make_layer(64,  layers[0], 1)
        self.layer2 = make_layer(128, layers[1], 2)
        self.layer3 = make_layer(256, layers[2], 2)
        self.layer4 = make_layer(512, layers[3], 2)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

        # gating 权重
        self.w_img = nn.Parameter(torch.tensor(1.0, device=device))
        self.w_text = nn.Parameter(torch.tensor(1.0, device=device))
        self.w_graph = nn.Parameter(torch.tensor(1.0, device=device))

        self.fc = nn.Linear(512 + t_dim + g_dim, num_classes)

    def forward(self, x, text_vec, graph_vec):
        # x: (B,1,H,W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgp(x).flatten(1)  # (B,512)
        B = x.size(0)

        img_f = x * self.w_img
        txt_f = text_vec.unsqueeze(0).expand(B, -1) * self.w_text
        gnn_f = graph_vec.unsqueeze(0).expand(B, -1) * self.w_graph

        fused = torch.cat([img_f, txt_f, gnn_f], dim=1)  # (B,512+t_dim+g_dim)
        logits = self.fc(fused)
        return logits


def build_model(num_classes, t_dim, g_dim, device):
    return ResNetFusionGated(
        BasicBlock, [2, 2, 2, 2],
        num_classes=num_classes,
        t_dim=t_dim,
        g_dim=g_dim,
        in_ch=1,
        device=device,
    ).to(device)


# =========================
# 主流程
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[Info] Using device: {device}")

    # ---------- 1) 读图像 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"图像目录不存在: {args.data_dir}")

    images, labels = [], []
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
                arr = np.array(img, dtype=np.float32) / 255.0  # (H,W)
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
        raise RuntimeError("未成功读取到任何图像样本。")

    images = np.stack(images)  # (N,H,W)
    labels = np.array(labels)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    print(f"[Info] 样本数={len(images)}, 类别数={num_classes}, 类别={list(le.classes_)}")

    # 划分
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_ds = ImageDataset(images[train_idx], y[train_idx])
    test_ds = ImageDataset(images[test_idx], y[test_idx])
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

    # ---------- 2) 文本先验 + BERT ----------
    if not os.path.isfile(args.text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{args.text_data_path}")
    df = pd.read_excel(args.text_data_path)

    col_names = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in col_names:
        if c not in df.columns:
            raise KeyError(f"Excel 中未找到列 '{c}'")
    if not (0 <= args.row_index < len(df)):
        raise IndexError(f"--row_index 超出范围 (0 ~ {len(df)-1})")

    combined_text = " ".join(str(df[c].iloc[args.row_index]) for c in col_names)
    print("[Info] 文本先验：", combined_text)

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    bert_model = BertModel.from_pretrained(args.bert_name).to(device)

    # 先全部冻结
    for p in bert_model.parameters():
        p.requires_grad = False

    # 解冻最后 N 层
    if args.unfreeze_last_n > 0:
        L = len(bert_model.encoder.layer)
        for i in range(L - args.unfreeze_last_n, L):
            for p in bert_model.encoder.layer[i].parameters():
                p.requires_grad = True
        bert_model.train()
    else:
        bert_model.eval()

    # 预先 tokenize，一次重用
    txt_inputs = tokenizer(combined_text, return_tensors="pt",
                           truncation=True, padding=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    # 若不微调 BERT，则预计算文本向量
    if args.unfreeze_last_n == 0:
        with torch.no_grad():
            txt_vec_fixed = bert_model(**txt_inputs).last_hidden_state[:, 0, :].squeeze(0)
        txt_vec_fixed = txt_vec_fixed.to(device)
        text_dim = txt_vec_fixed.numel()
        print(f"[Info] 文本向量（固定）维度: {text_dim}")
    else:
        # 动态计算，维度为 hidden_size
        text_dim = bert_model.config.hidden_size
        print(f"[Info] 文本向量（可微调）维度: {text_dim}")

    # ---------- 3) HeteroGNN Encoder（示例异质图） ----------
    hetero_enc = HeteroGNNEncoder(
        in_feats=args.hetero_in_dim,
        hid_feats=args.hetero_hidden_dim,
        out_feats=args.hetero_out_dim,
        device=device,
    ).to(device)

    with torch.no_grad():
        graph_data = {
            "type1": torch.randn(args.hetero_type1_nodes, args.hetero_in_dim),
            "type2": torch.randn(args.hetero_type2_nodes, args.hetero_in_dim),
        }
        graph_vec = hetero_enc(graph_data)  # (hetero_out_dim,)
        graph_vec = graph_vec.to(device)

    print(f"[Info] 图结构向量维度: {graph_vec.numel()}")

    if args.hetero_out_dim != text_dim:
        print(f"[Warn] hetero_out_dim({args.hetero_out_dim}) != text_dim({text_dim})，仍将直接拼接使用。")

    # ---------- 4) 构建 ResNet + Gating 模型 ----------
    model = build_model(
        num_classes=num_classes,
        t_dim=text_dim,
        g_dim=args.hetero_out_dim,
        device=device,
    )

    # ---------- 5) 优化器 & 调度器 ----------
    param_groups = [{"params": model.parameters(), "lr": args.lr_img}]
    if args.unfreeze_last_n > 0:
        L = len(bert_model.encoder.layer)
        for i in range(L - args.unfreeze_last_n, L):
            param_groups.append({
                "params": bert_model.encoder.layer[i].parameters(),
                "lr": args.lr_bert,
            })

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---------- 6) 训练 ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.unfreeze_last_n > 0:
            bert_model.train()
        total_loss = 0.0
        total = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 文本向量（根据是否微调决定是否每步重算）
            if args.unfreeze_last_n > 0:
                bert_out = bert_model(**txt_inputs)
                txt_vec = bert_out.last_hidden_state[:, 0, :].squeeze(0)
            else:
                txt_vec = txt_vec_fixed

            logits = model(imgs, txt_vec, graph_vec)
            loss = criterion(logits, lbls)

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total += bs

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        print(f"Epoch {epoch:02d}/{args.epochs} — loss: {avg_loss:.4f}")

    # ---------- 7) 测试 ----------
    model.eval()
    bert_model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            if args.unfreeze_last_n > 0:
                bert_out = bert_model(**txt_inputs)
                txt_vec = bert_out.last_hidden_state[:, 0, :].squeeze(0)
            else:
                txt_vec = txt_vec_fixed

            out = model(imgs, txt_vec, graph_vec)
            preds = out.argmax(dim=1)

            all_true.extend(lbls.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
    rec = recall_score(all_true, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    print(f"Test Accuracy      : {acc:.4f}")
    print(f"Precision (macro)  : {prec:.4f}")
    print(f"Recall    (macro)  : {rec:.4f}")
    print(f"F1-score  (macro)  : {f1:.4f}")

    # ---------- 8) 混淆矩阵 ----------
    if args.save_cm:
        cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.xticks(range(num_classes), le.classes_, rotation=45)
        plt.yticks(range(num_classes), le.classes_)
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(args.save_cm, dpi=150)
        plt.close()
        print(f"[Info] 混淆矩阵已保存到：{args.save_cm}")

    # ---------- 9) 保存模型 ----------
    if args.save_model:
        torch.save(
            {
                "model_state": model.state_dict(),
                "label_encoder_classes": le.classes_,
                "args": vars(args),
            },
            args.save_model,
        )
        print(f"[Info] 模型已保存到：{args.save_model}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
"""# 直接跑默认配置
python resnet_bert_hetero_gated.py

# 解冻 BERT 最后两层，保存模型和混淆矩阵，限制采样做快速实验
python resnet_bert_hetero_gated.py \
  --data_dir "../all" \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_cols "Type,List of Store Names" \
  --row_index 0 \
  --image_size 32 \
  --batch_size 32 \
  --epochs 20 \
  --lr_img 1e-3 \
  --lr_bert 5e-5 \
  --unfreeze_last_n 2 \
  --hetero_in_dim 16 \
  --hetero_hidden_dim 32 \
  --hetero_out_dim 768 \
  --hetero_type1_nodes 10 \
  --hetero_type2_nodes 15 \
  --device auto \
  --seed 42 \
  --limit 20000 \
  --save_model "resnet_bert_hetero_gated.pth" \
  --save_cm "cm_resnet_bert_hetero.png"
"""
