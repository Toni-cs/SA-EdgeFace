"""
QMUL-SurvFace 监控场景评测工具。

QMUL-SurvFace 提供监控人脸图像对与配对标签。本工具复用 LFW 的指标计算，
并额外支持按人脸像素宽度分档评测（监控小人脸专项）。

数据集目录约定（申请到数据后按此组织）:
    datasets/qmul_survface/
    ├── images/            # 监控人脸图片
    ├── pairs.txt          # 配对: img1 img2 is_same(0/1) [可选: width1 width2]
    └── ...

若 pairs.txt 无宽度列，则用检测 bbox 估计人脸宽度分档。

对应论文：Section 4.4（QMUL-SurvFace 结果）。
"""

import os
import numpy as np
from sklearn.metrics import roc_curve, auc

from .lfw_eval import evaluate_lfw


def load_qmul_pairs(pairs_path):
    """
    读取 QMUL-SurvFace pairs。
    支持格式: img1 img2 is_same [width1 width2]
    返回: list of (img1, img2, is_same, width1, width2)
    """
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img1, img2, label = parts[0], parts[1], int(parts[2])
            w1 = float(parts[3]) if len(parts) > 3 else None
            w2 = float(parts[4]) if len(parts) > 4 else None
            pairs.append((img1, img2, bool(label), w1, w2))
    return pairs


def evaluate_qmul(features, pairs, is_same_list):
    """复用 LFW 指标计算。"""
    return evaluate_lfw(features, pairs, is_same_list)


def evaluate_by_width(similarities, labels, widths, bins=(0, 32, 64, 96, 1e4)):
    """
    按人脸像素宽度分档计算准确率，突出监控小人脸增益。
    bins: (small, medium, large) 的宽度边界。
    """
    similarities = np.array(similarities)
    labels = np.array(labels)
    widths = np.array(widths, dtype=float)
    bin_names = ["small(<32px)", "medium(32-64)", "large(>64)"]
    results = {}
    for i in range(len(bins) - 1):
        mask = (widths >= bins[i]) & (widths < bins[i + 1])
        if mask.sum() < 2:
            results[bin_names[i]] = {"count": int(mask.sum()), "acc": None}
            continue
        s = similarities[mask]
        y = labels[mask]
        best_acc, best_t = 0, 0
        for t in np.linspace(s.min(), s.max(), 200):
            acc = np.mean((s >= t).astype(int) == y)
            if acc > best_acc:
                best_acc, best_t = acc, t
        results[bin_names[i]] = {"count": int(mask.sum()), "acc": best_acc, "threshold": best_t}
    return results


def print_qmul_results(results, by_width=None, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  QMUL-SurvFace Results: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy:       {results['accuracy']*100:.2f}%")
    print(f"  AUC:            {results['auc']:.4f}")
    print(f"  EER:            {results['eer']*100:.2f}%")
    for k, v in results.items():
        if k.startswith("tar@"):
            print(f"  {k}:       {v*100:.2f}%")
    if by_width:
        print(f"  --- by face width ---")
        for name, r in by_width.items():
            acc = f"{r['acc']*100:.2f}%" if r.get("acc") is not None else "N/A"
            print(f"  {name:<18} n={r['count']:<5} acc={acc}")
    print(f"{'='*55}\n")