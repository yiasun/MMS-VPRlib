# 文件名：svc_fusion.py
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser("BERT先验 + PCA + SVC（命令行可调参版）")
    # 数据路径
    p.add_argument("--text_data_path", type=str, default="Final Dataset-Texts.xlsx", help="Excel 文件路径")
    p.add_argument("--text_col", type=str, default="List of Store Names", help="Excel 中的文本列名")
    p.add_argument("--img_data_dir", type=str, default="../all", help="图像目录：子文件夹名即为类别")
    # BERT/设备
    p.add_argument("--bert_name", type=str, default="bert-base-chinese", help="BERT 模型名")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="BERT 计算设备")
    # 图像与PCA
    p.add_argument("--image_size", type=int, default=32, help="灰度缩放尺寸（image_size×image_size）")
    p.add_argument("--pca_components", type=int, default=128, help="PCA 降维维度")
    p.add_argument("--pca_whiten", action="store_true", help="PCA 使用 whiten=True")
    # 数据划分与复现
    p.add_argument("--test_size", type=float, default=0.3, help="测试集占比")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--limit", type=int, default=0, help="仅采样前 N 张图像做快速调试（0 表示全量）")
    # SVC 超参
    p.add_argument("--kernel", type=str, default="rbf", choices=["linear","poly","rbf","sigmoid","precomputed"], help="SVC kernel")
    p.add_argument("--C", type=float, default=1.0, help="SVC C")
    p.add_argument("--gamma", type=str, default="scale", help="SVC gamma（'scale'/'auto'或浮点数）")
    p.add_argument("--degree", type=int, default=3, help="SVC degree（poly）")
    p.add_argument("--coef0", type=float, default=0.0, help="SVC coef0（poly/sigmoid）")
    p.add_argument("--probability", action="store_true", help="启用概率输出（会变慢）")
    p.add_argument("--class_weight", type=str, default="", help="类别权重：'' 或 'balanced'")
    p.add_argument("--max_iter", type=int, default=-1, help="最大迭代步数，-1 为不设限")
    # 可选输出
    p.add_argument("--save_cm", type=str, default="", help="若非空，保存混淆矩阵到该路径（例如 cm_svc.png）")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    # 1) 文本先验：Excel 第一行文本 → BERT [CLS]
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
        text_hidden_state = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
        text_hidden_state_np = text_hidden_state.detach().cpu().numpy()
    print("文本 hidden state shape:", text_hidden_state_np.shape)

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

    # 5) 多模态特征拼接（图像降维 + 文本先验）
    def fuse_features(X, text_vector_np):
        n = X.shape[0]
        text_expanded = np.tile(text_vector_np, (n, 1))  # (n, 768)
        return np.concatenate([X, text_expanded], axis=1)

    fused_train = fuse_features(X_train_reduced, text_hidden_state_np)
    fused_test = fuse_features(X_test_reduced, text_hidden_state_np)
    print("融合后特征维度：", fused_train.shape[1])

    # 6) SVC 训练与评估
    gamma = args.gamma
    try:
        # 如果传了数字字符串，转成 float
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)
    except Exception:
        raise ValueError("--gamma 参数必须是 'scale'/'auto' 或可解析为浮点数的字符串")

    class_weight = None
    if args.class_weight.strip():
        if args.class_weight.strip() == "balanced":
            class_weight = "balanced"
        else:
            raise ValueError("--class_weight 仅支持 '' 或 'balanced'")

    svc_model = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=gamma,
        degree=args.degree,
        coef0=args.coef0,
        probability=args.probability,
        class_weight=class_weight,
        max_iter=args.max_iter,
        random_state=args.seed,
        decision_function_shape="ovr",
        break_ties=False,
    )
    print(f"[Info] SVC params: kernel={args.kernel}, C={args.C}, gamma={gamma}, degree={args.degree}, "
          f"coef0={args.coef0}, prob={args.probability}, class_weight={class_weight}, max_iter={args.max_iter}")

    svc_model.fit(fused_train, y_train)
    y_pred = svc_model.predict(fused_test)

    acc = accuracy_score(y_test, y_pred)
    print("测试集准确率：", acc)
    print("分类报告：")
    # 使用固定顺序的类别名，报告更稳定
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("多模态模型混淆矩阵 (SVC)")
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150)
        print(f"混淆矩阵已保存到：{args.save_cm}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
# 默认参数
python svc_fusion.py

# 常用调参
python svc_fusion.py \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --img_data_dir "../all" \
  --image_size 32 \
  --pca_components 128 \
  --pca_whiten \
  --test_size 0.3 \
  --seed 2025 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --kernel rbf \
  --C 2.0 \
  --gamma scale \
  --probability \
  --class_weight balanced \
  --max_iter 2000 \
  --save_cm "cm_svc.png"
'''