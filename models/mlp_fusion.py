# 文件名：mlp_fusion.py
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser("BERT先验 + PCA + MLP（命令行可调参版）")
    # 数据路径
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="Excel 文件路径")
    p.add_argument("--text_col", type=str, default="List of Store Names", help="Excel 中文本列名")
    p.add_argument("--img_data_dir", type=str, default="../all", help="图像根目录：子文件夹名即为类别")
    # BERT/设备
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="BERT 计算设备")
    # 图像与 PCA
    p.add_argument("--image_size", type=int, default=32, help="灰度缩放尺寸（image_size×image_size）")
    p.add_argument("--pca_components", type=int, default=128, help="PCA 降维维度")
    p.add_argument("--pca_whiten", action="store_true", help="PCA 使用 whiten=True")
    # 划分与复现
    p.add_argument("--test_size", type=float, default=0.2, help="测试集占比")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--limit", type=int, default=0, help="仅采样前 N 张图用于快速调试（0 为全量）")
    # 特征缩放
    p.add_argument("--scale", action="store_true", help="对融合特征做 StandardScaler（建议对 MLP 开启）")
    # MLP 超参
    p.add_argument("--hidden", type=str, default="256,128", help="隐藏层，如 '256,128' 或 '512'")
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "logistic"], help="激活函数")
    p.add_argument("--alpha", type=float, default=1e-4, help="L2 正则系数 alpha")
    p.add_argument("--lr_init", type=float, default=1e-3, help="学习率（learning_rate_init）")
    p.add_argument("--solver", type=str, default="adam", choices=["adam", "sgd", "lbfgs"], help="优化器")
    p.add_argument("--batch_size", type=str, default="auto", help="batch_size（int 或 'auto'）")
    p.add_argument("--max_iter", type=int, default=300, help="最大迭代轮数")
    p.add_argument("--early_stopping", action="store_true", help="启用早停")
    p.add_argument("--tol", type=float, default=1e-4, help="收敛容忍度 tol")
    # 输出
    p.add_argument("--save_cm", type=str, default="", help="若非空，保存混淆矩阵到该路径（如 cm_mlp.png）")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_hidden(s: str):
    # "256,128" -> (256,128)；"512" -> (512,)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(x) for x in parts)


def main(args):
    set_seed(args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    # 1) 文本先验：Excel 第一行 → BERT [CLS]
    if not os.path.isfile(args.text_data_path):
        raise FileNotFoundError(f"未找到 Excel：{args.text_data_path}")
    df_text = pd.read_excel(args.text_data_path)
    if args.text_col not in df_text.columns:
        raise KeyError(f"Excel 中未找到列 '{args.text_col}'")
    first_text = str(df_text[args.text_col].iloc[0])
    print("第一行文本：", first_text)

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    bert_model = BertModel.from_pretrained(args.bert_name).to(device)
    bert_model.eval()
    with torch.no_grad():
        inputs = tokenizer(first_text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        text_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
        text_vec_np = text_vec.detach().cpu().numpy()
    print("文本 hidden state shape:", text_vec_np.shape)

    # 2) 读取灰度图像，缩放并展平
    if not os.path.isdir(args.img_data_dir):
        raise FileNotFoundError(f"图像目录不存在：{args.img_data_dir}")

    image_features, labels = [], []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    class_dirs = [d for d in os.listdir(args.img_data_dir) if os.path.isdir(os.path.join(args.img_data_dir, d))]
    if len(class_dirs) == 0:
        raise RuntimeError("未在图像目录下发现任何子文件夹（类别）。")

    collected = 0
    for class_name in class_dirs:
        class_path = os.path.join(args.img_data_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(valid_ext):
                p = os.path.join(class_path, filename)
                try:
                    img = Image.open(p).convert('L').resize((args.image_size, args.image_size))
                    arr = np.array(img, dtype=np.float32).reshape(-1) / 255.0
                    image_features.append(arr)
                    labels.append(class_name)
                    collected += 1
                    if args.limit and collected >= args.limit:
                        break
                except Exception as e:
                    print(f"[警告] 读取失败，已跳过：{p}；错误：{e}")
        if args.limit and collected >= args.limit:
            break

    if len(image_features) == 0:
        raise RuntimeError("未成功读取到任何图像样本。")

    image_features = np.array(image_features, dtype=np.float32)
    labels = np.array(labels)
    classes = sorted(np.unique(labels).tolist())
    print("图像样本数：", image_features.shape[0])
    print("图像原始特征维度：", image_features.shape[1])
    print("类别：", classes)

    # 3) 划分训练/测试（分层）
    X_train, X_test, y_train, y_test = train_test_split(
        image_features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    print("训练样本数：", X_train.shape[0], "测试样本数：", X_test.shape[0])

    # 4) PCA 降维
    pca_kwargs = dict(n_components=args.pca_components, random_state=args.seed)
    if args.pca_whiten:
        pca_kwargs["whiten"] = True
    pca = PCA(**pca_kwargs)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    print("降维后训练集特征维度：", X_train_reduced.shape[1])

    # 5) 多模态特征：拼接（图像降维 + 文本先验）
    def fuse_features(X, text_np):
        n = X.shape[0]
        text_expanded = np.tile(text_np, (n, 1))   # (n, 768)
        return np.concatenate([X, text_expanded], axis=1)

    fused_train = fuse_features(X_train_reduced, text_vec_np)
    fused_test = fuse_features(X_test_reduced, text_vec_np)
    print("融合后特征维度：", fused_train.shape[1])

    # 6) （可选）标准化
    if args.scale:
        scaler = StandardScaler()
        fused_train = scaler.fit_transform(fused_train)
        fused_test = scaler.transform(fused_test)
        print("[Info] 已对融合特征进行 StandardScaler 标准化")

    # 7) 训练与评估 MLPClassifier
    hidden = parse_hidden(args.hidden)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=args.activation,
        alpha=args.alpha,
        learning_rate_init=args.lr_init,
        solver=args.solver,
        batch_size=(args.batch_size if args.batch_size == "auto" else int(args.batch_size)),
        max_iter=args.max_iter,
        early_stopping=args.early_stopping,
        tol=args.tol,
        random_state=args.seed,
        verbose=False
    )
    print(f"[Info] MLP params: hidden={hidden}, act={args.activation}, alpha={args.alpha}, "
          f"lr_init={args.lr_init}, solver={args.solver}, bs={args.batch_size}, "
          f"max_iter={args.max_iter}, early_stopping={args.early_stopping}, tol={args.tol}")

    mlp.fit(fused_train, y_train)
    y_pred = mlp.predict(fused_test)

    acc = accuracy_score(y_test, y_pred)
    print("测试集准确率：", acc)
    print("分类报告：")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("多模态模型混淆矩阵 (MLP)")
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150)
        print(f"混淆矩阵已保存到：{args.save_cm}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

