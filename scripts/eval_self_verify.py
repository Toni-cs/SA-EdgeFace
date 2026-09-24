"""
加载训练好的 EdgeFace，在 train_mini 上做 1:1 验证评测：
构造同/不同身份对，算 cosine 分布 + Accuracy + AUC，验证特征可分性。
"""

import os
import sys
import glob
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.edgeface import build_edgeface, DEFAULT_CONFIGS
from qad_face.utils import imread_cn
import cv2

CKPT = "weights/distill_mini/mini_x05_final.pt"
DATA = "datasets/train_mini"
CONFIG = "edgeface_x0_5_sab"


def extract_all(model, paths, device="cpu"):
    model.eval()
    feats = []
    with torch.no_grad():
        for p in paths:
            img = imread_cn(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
            t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
            emb = model.get_embedding(t)
            feats.append(emb.cpu().numpy().flatten())
    return np.array(feats)


def main():
    t0 = time.time()
    model = build_edgeface(DEFAULT_CONFIGS[CONFIG])
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=False)
    print(f"[load] {CKPT} {time.time()-t0:.1f}s", flush=True)

    id_dirs = sorted([d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))])
    paths, labels = [], []
    for i, d in enumerate(id_dirs):
        for p in sorted(glob.glob(os.path.join(DATA, d, "*.jpg"))):
            paths.append(p)
            labels.append(i)
    print(f"[data] {len(paths)} imgs, {len(id_dirs)} ids", flush=True)

    feats = extract_all(model, paths)
    norms = np.linalg.norm(feats, axis=1)
    print(f"[feat] {feats.shape}, norm range [{norms.min():.3f}, {norms.max():.3f}]", flush=True)

    labels = np.array(labels)
    sims, same = [], []
    n = len(paths)
    step = max(1, n // 200)
    for i in range(0, n, step):
        for j in range(i + 1, min(i + 50, n)):
            sim = np.dot(feats[i], feats[j]) / (norms[i] * norms[j] + 1e-8)
            sims.append(sim)
            same.append(1 if labels[i] == labels[j] else 0)
    sims = np.array(sims)
    same = np.array(same)

    auc = roc_auc_score(same, sims)
    best_acc, best_t = 0, 0
    for t in np.linspace(sims.min(), sims.max(), 500):
        acc = np.mean((sims >= t).astype(int) == same)
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"\n{'='*50}")
    print(f"  Self-verification on train_mini ({CONFIG})")
    print(f"{'='*50}")
    print(f"  Pairs: {len(same)} (same={same.sum()}, diff={len(same)-same.sum()})")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {best_acc*100:.2f}% @ thresh={best_t:.4f}")
    print(f"  Same-pair sim:  mean={sims[same==1].mean():.4f}")
    print(f"  Diff-pair sim:  mean={sims[same==0].mean():.4f}")
    print(f"  Separation:     {sims[same==1].mean()-sims[same==0].mean():.4f}")
    print(f"{'='*50}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()