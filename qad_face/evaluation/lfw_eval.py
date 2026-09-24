"""
LFW (Labeled Faces in the Wild) 评测工具。

支持标准 LFW 评测协议：读取 pairs.txt，计算人脸对相似度，输出准确率、AUC、TAR@FAR。
"""

import os
import numpy as np
from sklearn.metrics import roc_curve, auc


def load_pairs(pairs_path):
    """
    读取 LFW pairs.txt 文件。

    返回:
        pairs: list of tuples (name1, idx1, name2, idx2, is_same)
    """
    pairs = []
    with open(pairs_path, "r") as f:
        lines = f.readlines()
    # 第一行是 header: 10 300 (10折, 每折300对)
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            # 同一个人: name idx1 idx2
            name, idx1, idx2 = parts
            pairs.append((name, int(idx1), name, int(idx2), True))
        elif len(parts) == 4:
            # 不同人: name1 idx1 name2 idx2
            name1, idx1, name2, idx2 = parts
            pairs.append((name1, int(idx1), name2, int(idx2), False))
    return pairs


def get_image_path(lfw_dir, name, idx):
    """构造 LFW 图片路径: lfw_dir/name/name_0001.jpg"""
    return os.path.join(lfw_dir, name, f"{name}_{idx:04d}.jpg")


def evaluate_lfw(features, pairs, is_same_list):
    """
    给定特征和标签，计算 LFW 指标。

    Args:
        features: np.array, shape (N, 512), 所有人脸的特征（按 pairs 顺序，每对两个特征）
        pairs: list, 从 load_pairs 返回
        is_same_list: list of bool, 每对是否同一人

    Returns:
        dict: 包含 accuracy, auc, tar_far1e-3, eer 等指标
    """
    similarities = []
    labels = []
    for i in range(len(pairs)):
        feat1 = features[2 * i]
        feat2 = features[2 * i + 1]
        # 余弦相似度
        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
        similarities.append(sim)
        labels.append(1 if is_same_list[i] else 0)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # 准确率（最优阈值）
    best_acc = 0
    best_thresh = 0
    for thresh in thresholds:
        pred = (similarities >= thresh).astype(int)
        acc = np.mean(pred == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # TAR@FAR
    tar_at_far = {}
    for far_target in [1e-3, 1e-4, 1e-6]:
        idx = np.searchsorted(fpr, far_target)
        if idx < len(tpr):
            tar_at_far[f"tar@{far_target}"] = tpr[idx]
        else:
            tar_at_far[f"tar@{far_target}"] = 0.0

    # EER (等错误率)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return {
        "accuracy": best_acc,
        "best_threshold": best_thresh,
        "auc": roc_auc,
        "eer": eer,
        **tar_at_far,
        "num_pairs": len(pairs),
        "num_same": int(np.sum(labels)),
        "num_diff": int(np.sum(1 - labels)),
    }


def print_results(results, model_name="Model"):
    """格式化打印评测结果。"""
    print(f"\n{'='*50}")
    print(f"  LFW Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Total pairs:    {results['num_pairs']}")
    print(f"  Same pairs:     {results['num_same']}")
    print(f"  Diff pairs:     {results['num_diff']}")
    print(f"  Accuracy:       {results['accuracy']*100:.2f}%")
    print(f"  AUC:            {results['auc']:.4f}")
    print(f"  EER:            {results['eer']*100:.2f}%")
    print(f"  Best threshold: {results['best_threshold']:.4f}")
    for key, val in results.items():
        if key.startswith("tar@"):
            print(f"  {key}:       {val*100:.2f}%")
    print(f"{'='*50}\n")
