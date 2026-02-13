# 文件名：gcn_textplugin.py
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from transformers import BertTokenizer, BertModel


# =========================
# 命令行参数
# =========================
def parse_args():
    p = argparse.ArgumentParser("GCN + BERT 文本先验（命令行可调参版）")
    # 图像数据
    p.add_argument("--data_dir", type=str, default="../all", help="图像根目录：子文件夹名即类别")
    p.add_argument("--image_size", type=int, default=32, help="灰度缩放尺寸 (image_size x image_size)")
    p.add_argument("--limit", type=int, default=0, help="仅使用前 N 张图做实验（0 表示全量）")
    p.add_argument("--test_size", type=float, default=0.2, help="测试集占比（节点划分）")
    # 图构建
    p.add_argument("--knn_k", type=int, default=10, help="KNN 中的 k")
    p.add_argument("--knn_metric", type=str, default="euclidean",
                   choices=["euclidean", "cosine", "manhattan"],
                   help="KNN 距离度量")
    p.add_argument("--no_bidirectional", action="store_true",
                   help="不强制加双向边（默认加双向边）")
    # 文本 / BERT
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="Excel 路径")
    p.add_argument("--text_col", type=str, default="List of Store Names", help="用于先验的文本列名")
    p.add_argument("--row_index", type=int, default=0, help="使用该行的文本（从 0 开始）")
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    # 模型 / 训练
    p.add_argument("--hidden_dim", type=int, default=64, help="GCN 隐藏维度")
    p.add_argument("--epochs", type=int, default=200, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减")
    # 设备 / 复现
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="设备选择")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    # 输出
    p.add_argument("--save_model", type=str, default="",
                   help="若非空，则保存模型到该路径（.pth）")
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
# 模型定义
# =========================
class GCNPlugin(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, text_dim=768):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels + text_dim, out_channels)

    def forward(self, data: Data, text_vec: torch.Tensor):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # x: [num_nodes, hidden_channels]
        num_nodes = x.size(0)
        txt = text_vec.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, text_dim]
        fused = torch.cat([x, txt], dim=1)
        logits = self.classifier(fused)  # [num_nodes, out_channels]
        return logits


# =========================
# 主流程
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[Info] Using device: {device}")

    # ---------- 1) 加载图像数据 ----------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"图像目录不存在: {args.data_dir}")

    X_list, y_list = [], []
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
            img_path = os.path.join(cls_path, fn)
            try:
                img = Image.open(img_path).convert("L")
                img = img.resize((args.image_size, args.image_size))
                arr = np.array(img, dtype=np.float32).reshape(-1) / 255.0  # 展平
                X_list.append(arr)
                y_list.append(cls)
                count += 1
                if args.limit and count >= args.limit:
                    break
            except Exception as e:
                print(f"[警告] 读取失败，跳过 {img_path}，错误：{e}")
        if args.limit and count >= args.limit:
            break

    if len(X_list) == 0:
        raise RuntimeError("未成功读取到任何图像样本。")

    X = np.stack(X_list).astype("float32")  # [N, D]
    y = np.array(y_list)
    print(f"[Info] 加载图像数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_nodes, in_channels = X.shape
    num_classes = len(le.classes_)
    print(f"[Info] 类别数: {num_classes}, 类别: {list(le.classes_)}")

    # ---------- 2) KNN 构图 ----------
    print(f"[Info] 构图中: k={args.knn_k}, metric={args.knn_metric}")
    nbrs = NearestNeighbors(n_neighbors=args.knn_k,
                            algorithm="auto",
                            metric=args.knn_metric)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    edges = []
    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue
            edges.append([i, j])
            # 默认加反向边，除非显式关闭
            if not args.no_bidirectional:
                edges.append([j, i])

    if len(edges) == 0:
        raise RuntimeError("KNN 未生成任何边，请检查数据或参数。")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    print(f"[Info] 图中边数: {edge_index.size(1)}")

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

    # ---------- 3) 节点划分 train/test ----------
    all_idx = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    print(f"[Info] 训练节点数: {int(train_mask.sum())}, 测试节点数: {int(test_mask.sum())}")

    # ---------- 4) 文本先验 (BERT [CLS]) ----------
    if not os.path.isfile(args.text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{args.text_data_path}")
    df_text = pd.read_excel(args.text_data_path)

    if args.text_col not in df_text.columns:
        raise KeyError(f"Excel 中未找到列 '{args.text_col}'")
    if not (0 <= args.row_index < len(df_text)):
        raise IndexError(f"--row_index 超出范围（0 ~ {len(df_text)-1}）")

    first_text = str(df_text[args.text_col].iloc[args.row_index])
    print("[Info] 文本先验内容：", first_text)

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    bert_model = BertModel.from_pretrained(args.bert_name).to(device)
    bert_model.eval()
    with torch.no_grad():
        inputs = tokenizer(first_text, return_tensors="pt",
                           truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        text_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]

    text_dim = text_vec.numel()
    print(f"[Info] 文本向量维度: {text_dim}")

    # ---------- 5) 构建模型 ----------
    model = GCNPlugin(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=num_classes,
        text_dim=text_dim
    ).to(device)

    data = data.to(device)
    text_vec = text_vec.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ---------- 6) 训练 ----------
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        out = model(data, text_vec)  # [N, num_classes]
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == args.epochs or epoch % 20 == 0:
            with torch.no_grad():
                pred_train = out.argmax(dim=1)
                train_acc = (pred_train[data.train_mask] ==
                             data.y[data.train_mask]).float().mean().item()
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | TrainAcc {train_acc:.4f}")

    # ---------- 7) 评估 ----------
    model.eval()
    with torch.no_grad():
        out = model(data, text_vec)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Precision(mac): {prec:.4f}")
    print(f"Recall   (mac): {rec:.4f}")
    print(f"F1-score (mac): {f1:.4f}")

    # ---------- 8) 保存模型（可选） ----------
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
