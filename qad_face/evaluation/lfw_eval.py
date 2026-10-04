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

    使用标准 LFW 10-fold 交叉验证协议：
    - 6000 对分 10 折（每折 300 same + 300 diff）
    - 每折：在其余 9 折上搜最优阈值，在本折上评测
    - accuracy = 10 折均值

    AUC/EER/TAR@FAR 在全量 6000 对上计算（阈值无关指标）。

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
        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
        similarities.append(sim)
        labels.append(1 if is_same_list[i] else 0)

    similarities = np.array(similarities)
    labels = np.array(labels)
    n_pairs = len(pairs)

    # ROC 曲线 (全量，阈值无关)
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # 10-fold 交叉验证 accuracy (标准 LFW 协议)
    n_same = int(np.sum(labels))
    n_diff = n_pairs - n_same
    fold_size_same = n_same // 10
    fold_size_diff = n_diff // 10

    same_indices = np.where(labels == 1)[0]
    diff_indices = np.where(labels == 0)[0]

    fold_accs = []
    fold_thresholds = []
    for fold in range(10):
        test_same = same_indices[fold * fold_size_same:(fold + 1) * fold_size_same]
        test_diff = diff_indices[fold * fold_size_diff:(fold + 1) * fold_size_diff]
        test_idx = np.concatenate([test_same, test_diff])
        val_idx = np.setdiff1d(np.arange(n_pairs), test_idx)

        val_sims = similarities[val_idx]
        val_labels = labels[val_idx]
        test_sims = similarities[test_idx]
        test_labels = labels[test_idx]

        val_fpr, val_tpr, val_thresh = roc_curve(val_labels, val_sims)
        best_acc_val = 0
        best_t = 0.0
        for t in val_thresh:
            pred = (val_sims >= t).astype(int)
            acc = np.mean(pred == val_labels)
            if acc > best_acc_val:
                best_acc_val = acc
                best_t = t

        pred_test = (test_sims >= best_t).astype(int)
        fold_acc = np.mean(pred_test == test_labels)
        fold_accs.append(fold_acc)
        fold_thresholds.append(best_t)

    cv_accuracy = np.mean(fold_accs)
    cv_std = np.std(fold_accs)

    # Oracle accuracy (全量搜阈值，仅作参考，不作为报告值)
    best_acc_oracle = 0
    best_thresh_oracle = 0
    for thresh in thresholds:
        pred = (similarities >= thresh).astype(int)
        acc = np.mean(pred == labels)
        if acc > best_acc_oracle:
            best_acc_oracle = acc
            best_thresh_oracle = thresh

    # TAR@FAR (全量 ROC)
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
        "accuracy": cv_accuracy,
        "accuracy_std": cv_std,
        "accuracy_oracle": best_acc_oracle,
        "best_threshold": np.mean(fold_thresholds),
        "auc": roc_auc,
        "eer": eer,
        **tar_at_far,
        "num_pairs": n_pairs,
        "num_same": n_same,
        "num_diff": n_diff,
    }


def print_results(results, model_name="Model"):
    """格式化打印评测结果。"""
    print(f"\n{'='*50}")
    print(f"  LFW Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Total pairs:    {results['num_pairs']}")
    print(f"  Same pairs:     {results['num_same']}")
    print(f"  Diff pairs:     {results['num_diff']}")
    print(f"  Accuracy (10-fold CV): {results['accuracy']*100:.2f}% +/- {results.get('accuracy_std',0)*100:.2f}%")
    print(f"  Accuracy (oracle):     {results.get('accuracy_oracle',0)*100:.2f}%  [参考，非报告值]")
    print(f"  AUC:            {results['auc']:.4f}")
    print(f"  EER:            {results['eer']*100:.2f}%")
    print(f"  Best threshold: {results['best_threshold']:.4f}")
    for key, val in results.items():
        if key.startswith("tar@"):
            print(f"  {key}:       {val*100:.2f}%")
    print(f"{'='*50}\n")
