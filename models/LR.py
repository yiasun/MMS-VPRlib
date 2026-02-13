
import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 命令行参数
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal fusion from CLI")
    parser.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx",
                        help="Excel 文件路径，需含列 'List of Store Names'")
    parser.add_argument("--img_data_dir", type=str, default="../all",
                        help="图像数据根目录，每个子文件夹为一个类别")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--test_size", type=float, default=0.5, help="测试集占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--image_size", type=int, default=32, help="图像缩放为 image_size × image_size（灰度）")
    parser.add_argument("--hidden_dim", type=int, default=128, help="图像分支隐藏维度")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="计算设备：auto/cpu/cuda")
    parser.add_argument("--model_out", type=str, default="",
                        help="若非空，则保存模型权重到该路径（.pth）")
    parser.add_argument("--save_cm", type=str, default="",
                        help="若非空，则将混淆矩阵保存至该路径（例如 cm.png），并不弹出窗口")
    return parser.parse_args()


# -----------------------------
# 数据集
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, features, labels, classes):
        self.features = features.astype(np.float32)  # (N, D)
        self.labels = labels
        self.class2idx = {cls: i for i, cls in enumerate(classes)}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])           # float32
        y = torch.tensor(self.class2idx[self.labels[idx]], dtype=torch.long)
        return x, y


# -----------------------------
# 模型
# -----------------------------
class MultiModalFusion(nn.Module):
    def __init__(self, image_input_dim=1024, image_hidden_dim=128, text_dim=768, num_classes=2):
        super(MultiModalFusion, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(image_input_dim, image_hidden_dim),
            nn.ReLU()
        )
        fusion_dim = image_hidden_dim + text_dim
        self.decoder = nn.Linear(fusion_dim, num_classes)

    def forward(self, x, text_vec):
        # x: [B, image_input_dim]
        # text_vec: [text_dim]
        img_hidden = self.image_branch(x)                          # [B, image_hidden_dim]
        batch_size = x.size(0)
        text_expanded = text_vec.unsqueeze(0).expand(batch_size, -1)  # [B, text_dim]
        fusion = torch.cat([img_hidden, text_expanded], dim=1)     # [B, fusion_dim]
        logits = self.decoder(fusion)                               # [B, num_classes]
        return logits


# -----------------------------
# 主流程
# -----------------------------
def main(args):
    # 设备与随机性
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1) 文本先验：Excel 第一行文本 → BERT [CLS] 向量
    text_data_path = args.text_data_path
    if not os.path.isfile(text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{text_data_path}")
    df_text = pd.read_excel(text_data_path)
    if 'List of Store Names' not in df_text.columns:
        raise KeyError("Excel 中未找到列 'List of Store Names'")
    first_text = str(df_text['List of Store Names'].iloc[0])
    print("第一行文本：", first_text)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    bert_model = BertModel.from_pretrained("bert-base-chinese").to(device)
    bert_model.eval()
    with torch.no_grad():
        inputs = tokenizer(first_text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        text_hidden_state = outputs.last_hidden_state[:, 0, :].squeeze(0)  # shape: (768,)
    print("文本 hidden state shape:", tuple(text_hidden_state.shape))

    # 2) 图像数据：子目录即为类别，灰度，缩放到 image_size×image_size 并展平
    img_data_dir = args.img_data_dir
    if not os.path.isdir(img_data_dir):
        raise FileNotFoundError(f"图像目录不存在：{img_data_dir}")

    image_features = []
    img_labels = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    class_dirs = [d for d in os.listdir(img_data_dir) if os.path.isdir(os.path.join(img_data_dir, d))]
    if len(class_dirs) == 0:
        raise RuntimeError("未在图像目录下发现任何子文件夹（类别）。")

    for class_name in class_dirs:
        class_path = os.path.join(img_data_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(valid_ext):
                p = os.path.join(class_path, filename)
                try:
                    img = Image.open(p)
                    img = img.convert('L')  # 保持与原脚本一致：灰度
                    img = img.resize((args.image_size, args.image_size))
                    arr = np.array(img, dtype=np.float32).flatten() / 255.0  # [image_size*image_size]
                    image_features.append(arr)
                    img_labels.append(class_name)
                except Exception as e:
                    print(f"[警告] 读取失败，已跳过：{p}；错误：{e}")

    if len(image_features) == 0:
        raise RuntimeError("未成功读取到任何图像样本。")

    image_features = np.array(image_features, dtype=np.float32)  # (N, D)
    img_labels = np.array(img_labels)
    classes = sorted(np.unique(img_labels))
    print("图像样本数:", image_features.shape[0])
    print("图像特征维度:", image_features.shape[1])
    print("类别：", classes)

    # 3) 构造 Dataset 与 DataLoader
    X_train, X_test, y_train, y_test = train_test_split(
        image_features, img_labels, test_size=args.test_size, random_state=args.seed, stratify=img_labels
    )
    train_dataset = ImageDataset(X_train, y_train, classes)
    test_dataset = ImageDataset(X_test, y_test, classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("训练样本数:", len(train_dataset), "测试样本数:", len(test_dataset))

    # 4) 定义模型（输入维度需与展平后的长度一致）
    image_input_dim = args.image_size * args.image_size  # 灰度展平
    num_classes = len(classes)
    model = MultiModalFusion(
        image_input_dim=image_input_dim,
        image_hidden_dim=args.hidden_dim,
        text_dim=768,
        num_classes=num_classes
    ).to(device)
    text_hidden_state = text_hidden_state.to(device)

    # 5) 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, text_hidden_state)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 6) 评估
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x, text_hidden_state)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    print("测试集准确率:", acc)
    print("分类报告:")
    print(classification_report(all_targets, all_preds, target_names=classes))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("多模态模型混淆矩阵")
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150)
        print(f"混淆矩阵已保存到：{args.save_cm}")
    else:
        plt.show()

    # 可选：保存模型
    if args.model_out:
        torch.save(model.state_dict(), args.model_out)
        print(f"模型权重已保存到：{args.model_out}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
