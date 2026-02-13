# 文件名：resnet_bert_fusion.py
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt


# -----------------------------
# 命令行参数
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("ResNet + BERT文本先验 融合分类（命令行可调参版）")
    # 数据与文本
    p.add_argument("--data_dir", type=str, default="../all", help="图像根目录，子文件夹名即类别")
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="Excel 路径")
    p.add_argument("--text_cols", type=str, default="Type,List of Store Names",
                   help="用于拼接文本的列名，用英文逗号分隔，按顺序拼接")
    p.add_argument("--row_index", type=int, default=0, help="取哪一行文本（默认第0行）")
    # 图像与Loader
    p.add_argument("--image_size", type=int, default=32, help="图像缩放边长（灰度）")
    p.add_argument("--batch_size", type=int, default=32, help="批大小")
    p.add_argument("--test_size", type=float, default=0.2, help="测试集占比")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--limit", type=int, default=0, help="仅采样前N张图用于调试（0为全量）")
    # 训练
    p.add_argument("--epochs", type=int, default=20, help="训练轮数")
    p.add_argument("--lr_img", type=float, default=1e-3, help="模型参数学习率")
    p.add_argument("--lr_bert", type=float, default=5e-5, help="BERT学习率")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    p.add_argument("--amp", action="store_true", help="启用混合精度训练")
    # BERT
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="设备")
    p.add_argument("--unfreeze_last_n", type=int, default=2, help="解冻BERT最后N层进行微调（0则全冻结）")
    # 复现与输出
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--save_model", type=str, default="", help="若非空，保存模型权重到该路径（.pth）")
    p.add_argument("--save_cm", type=str, default="", help="若非空，保存测试集混淆矩阵到该路径（.png）")
    return p.parse_args()


# -----------------------------
# 工具
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# -----------------------------
# 数据集
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # (N,H,W) float32 in [0,1]
        self.labels = labels  # int
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        x = self.images[idx][None, ...]  # (1,H,W)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# -----------------------------
# 模型
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.down = None
        if stride != 1 or in_planes != planes:
            self.down = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down:
            idt = self.down(x)
        return self.relu(out + idt)

class ResNetFusionText(nn.Module):
    def __init__(self, block, layers, num_classes, in_ch=1, text_dim=768, device=None):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        def mk(planes, cnt, stride):
            seq = [block(self.in_planes, planes, stride)]
            self.in_planes = planes * block.expansion
            for _ in range(1, cnt):
                seq.append(block(self.in_planes, planes))
            return nn.Sequential(*seq)

        self.layer1 = mk(64,  layers[0], 1)
        self.layer2 = mk(128, layers[1], 2)
        self.layer3 = mk(256, layers[2], 2)
        self.layer4 = mk(512, layers[3], 2)
        self.avgp   = nn.AdaptiveAvgPool2d((1,1))

        self.w_img  = nn.Parameter(torch.tensor(1.0, device=device))
        self.w_text = nn.Parameter(torch.tensor(1.0, device=device))
        self.fc = nn.Linear(512 + text_dim, num_classes)

    def forward(self, x, text_vec):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgp(x).flatten(1)
        B = x.size(0)
        img_f = x * self.w_img
        txt_f = text_vec.unsqueeze(0).expand(B, -1) * self.w_text   # same text vec for batch
        return self.fc(torch.cat([img_f, txt_f], dim=1))


def build_model(num_classes, device):
    return ResNetFusionText(BasicBlock, [2,2,2,2], num_classes, in_ch=1, text_dim=768, device=device)


def select_device(arg_device: str):
    if arg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg_device)


def prepare_bert(bert_name: str, device, unfreeze_last_n: int):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert = BertModel.from_pretrained(bert_name).to(device)

    # 先全部冻结
    for p in bert.parameters():
        p.requires_grad = False

    # 解冻最后N层encoder
    if unfreeze_last_n > 0:
        L = len(bert.encoder.layer)
        for i in range(L - unfreeze_last_n, L):
            for p in bert.encoder.layer[i].parameters():
                p.requires_grad = True

    return tokenizer, bert


def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[Info] device: {device}")

    # -----------------------------
    # 1) 读取图像与标签
    # -----------------------------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"图像目录不存在：{args.data_dir}")

    images, labels = [], []
    classes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    classes.sort()
    valid_ext = (".jpg",".jpeg",".png",".bmp")
    collected = 0
    for cls in classes:
        cls_path = os.path.join(args.data_dir, cls)
        for fn in os.listdir(cls_path):
            if fn.lower().endswith(valid_ext):
                try:
                    img = Image.open(os.path.join(cls_path, fn)).convert("L")
                    img = img.resize((args.image_size, args.image_size))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    images.append(arr); labels.append(cls)
                    collected += 1
                    if args.limit and collected >= args.limit:
                        break
                except Exception as e:
                    print(f"[警告] 读取失败，已跳过：{os.path.join(cls_path, fn)}; 错误：{e}")
        if args.limit and collected >= args.limit:
            break

    if len(images) == 0:
        raise RuntimeError("未成功读取到任何图像样本。")

    images = np.stack(images)  # (N,H,W)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"[Info] 样本数={len(images)}, 类别数={num_classes}")

    # 划分
    idx = np.arange(len(images))
    tidx, vidx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y)
    train_ds = ImageDataset(images[tidx], y[tidx])
    test_ds  = ImageDataset(images[vidx], y[vidx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # -----------------------------
    # 2) 文本数据与BERT
    # -----------------------------
    if not os.path.isfile(args.text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{args.text_data_path}")
    df = pd.read_excel(args.text_data_path)

    col_names = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for col in col_names:
        if col not in df.columns:
            raise KeyError(f"Excel 中未找到列 '{col}'")

    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"--row_index 超出范围（0~{len(df)-1}）")

    combined_text = " ".join(str(df[col].iloc[args.row_index]) for col in col_names)
    print("[Info] 文本：", combined_text)

    tokenizer, bert_model = prepare_bert(args.bert_name, device, args.unfreeze_last_n)
    # 预tokenize（每个batch都用同一段文本，但BERT参数在训练阶段可更新）
    txt_inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    # -----------------------------
    # 3) 模型
    # -----------------------------
    model = build_model(num_classes, device).to(device)

    # 优化器（模型 + BERT最后N层）
    param_groups = [{"params": model.parameters(), "lr": args.lr_img}]
    if args.unfreeze_last_n > 0:
        L = len(bert_model.encoder.layer)
        for i in range(L - args.unfreeze_last_n, L):
            param_groups.append({"params": bert_model.encoder.layer[i].parameters(), "lr": args.lr_bert})
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    # -----------------------------
    # 4) 训练
    # -----------------------------
    for epoch in range(1, args.epochs+1):
        model.train(); bert_model.train()  # 若unfreeze_last_n=0，BERT仍在eval态也OK
        total_loss, total, correct = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    text_vec = bert_model(**txt_inputs).last_hidden_state[:,0,:].squeeze(0)  # (768,)
                    logits = model(imgs, text_vec)
                    loss = criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                text_vec = bert_model(**txt_inputs).last_hidden_state[:,0,:].squeeze(0)
                logits = model(imgs, text_vec)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += imgs.size(0)

        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {total_loss/total:.4f} | acc {correct/total:.4f}")

    # -----------------------------
    # 5) 测试
    # -----------------------------
    model.eval(); bert_model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            if args.amp and device.type=="cuda":
                with torch.cuda.amp.autocast():
                    text_vec = bert_model(**txt_inputs).last_hidden_state[:,0,:].squeeze(0)
                    preds = model(imgs, text_vec).argmax(dim=1)
            else:
                text_vec = bert_model(**txt_inputs).last_hidden_state[:,0,:].squeeze(0)
                preds = model(imgs, text_vec).argmax(dim=1)

            all_true.extend(lbls.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, average='macro', zero_division=0)
    rec  = recall_score(all_true, all_pred, average='macro', zero_division=0)
    f1   = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall    (macro): {rec:.4f}")
    print(f"F1-score  (macro): {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.xticks(range(num_classes), le.classes_, rotation=45)
    plt.yticks(range(num_classes), le.classes_)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150)
        print(f"[Info] 混淆矩阵已保存到：{args.save_cm}")
    else:
        plt.show()

    # 保存模型
    if args.save_model:
        torch.save({
            "model_state": model.state_dict(),
            "label_encoder_classes": le.classes_,
            "args": vars(args)
        }, args.save_model)
        print(f"[Info] 模型已保存到：{args.save_model}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
# 默认参数
python resnet_bert_fusion.py

# 常用调参：解冻BERT最后两层、混合精度、保存模型与混淆矩阵、限制读取2万张图快速调试
python resnet_bert_fusion.py \
  --data_dir "../all" \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_cols "Type,List of Store Names" \
  --row_index 0 \
  --image_size 32 \
  --batch_size 64 \
  --test_size 0.2 \
  --epochs 20 \
  --lr_img 1e-3 \
  --lr_bert 5e-5 \
  --weight_decay 1e-2 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --unfreeze_last_n 2 \
  --amp \
  --seed 42 \
  --limit 20000 \
  --save_model "resnet_bert_fusion.pth" \
  --save_cm "cm_resnet_bert.png"
"""