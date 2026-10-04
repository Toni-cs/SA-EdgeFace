"""
生成 LFW ROC 曲线图（低 FAR 区域放大），用于论文 Figure 2。
对 C0-C4 五个配置提取 LFW pair 相似度，画 ROC 曲线。
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path

plt.rcParams.update({"font.size": 12, "font.family": "serif", "figure.dpi": 150})

LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"
SAVE_DIR = "weights/ablation_casia"
OUT_PDF = "paper/figs/roc_lfw.pdf"
OUT_PNG = "paper/figs/roc_lfw.png"

CONFIGS = [
    ("cfg0", "x05_baseline",    "C0 baseline",      "#888888"),

    ("cfg2", "x05_sab_spatial", "C2 +spatial",      "#DD8452"),
    ("cfg3", "x05_sab_full",    "C3 +full SAB",     "#55A868"),
    ("cfg4", "x05_sab_full",    "C4 +distill",      "#C44E52"),
]


def load_model(ckpt_path, cfg_name, device):
    model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    return model


def preload_images(paths):
    imgs = {}
    for p in paths:
        if os.path.exists(p):
            img = imread_cn(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                imgs[p] = np.transpose(img, (2, 0, 1))
    return imgs


def batch_extract(model, paths, device, imgs_cache, bs=256):
    cache = {}
    valid_idx = []
    valid_imgs = []
    for i, p in enumerate(paths):
        if p in imgs_cache:
            valid_idx.append(i)
            valid_imgs.append(imgs_cache[p])
        else:
            cache[p] = np.zeros(512, dtype=np.float32)

    valid_idx = np.array(valid_idx)
    feats = np.zeros((len(valid_imgs), 512), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(valid_imgs), bs):
            e = min(s + bs, len(valid_imgs))
            batch = np.array(valid_imgs[s:e], dtype=np.float32)
            t = torch.from_numpy(batch).to(device)
            f = model.get_embedding(t).cpu().numpy()
            feats[s:e] = f[:, :512]

    for j, i in enumerate(valid_idx):
        cache[paths[i]] = feats[j]
    return cache


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = load_pairs(PAIRS_PATH)
    print(f"LFW pairs: {len(pairs)}")

    needed = set()
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(LFW_DIR, name1, idx1))
        needed.add(get_image_path(LFW_DIR, name2, idx2))
    needed = sorted(needed)
    print(f"Unique images: {len(needed)}")

    print("Preloading images...")
    imgs_cache = preload_images(needed)
    print(f"  loaded {len(imgs_cache)} images")

    is_same_list = []
    for name1, idx1, name2, idx2, is_same in pairs:
        is_same_list.append(is_same)
    labels = np.array([1 if s else 0 for s in is_same_list])

    all_sims = {}

    for save_name, cfg_name, desc, color in CONFIGS:
        ckpt_path = os.path.join(SAVE_DIR, f"{save_name}_final.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP {desc}: {ckpt_path} not found")
            continue
        print(f"\n  {desc} ({save_name})")

        model = load_model(ckpt_path, cfg_name, device)
        cache = batch_extract(model, needed, device, imgs_cache)
        print(f"    features extracted")

        sims = []
        for name1, idx1, name2, idx2, is_same in pairs:
            p1 = get_image_path(LFW_DIR, name1, idx1)
            p2 = get_image_path(LFW_DIR, name2, idx2)
            f1, f2 = cache[p1], cache[p2]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            sims.append(sim)
        all_sims[desc] = (np.array(sims), color)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(7, 5))

    for desc, (sims, color) in all_sims.items():
        fpr, tpr, _ = roc_curve(labels, sims)
        ax.plot(fpr, tpr, label=desc, color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")
    ax.set_title("LFW ROC — Low-FAR Region")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    print(f"\nSaved: {OUT_PDF}")

    print(f"\n{'Config':<20} {'TAR@1e-4':>10} {'TAR@1e-3':>10} {'TAR@1e-2':>10}")
    print("-" * 52)
    for desc, (sims, color) in all_sims.items():
        fpr, tpr, _ = roc_curve(labels, sims)
        for far_t, tag in [(1e-4, "1e-4"), (1e-3, "1e-3"), (1e-2, "1e-2")]:
            idx = np.searchsorted(fpr, far_t)
            tar = tpr[idx] if idx < len(tpr) else 0.0
            if tag == "1e-4":
                t4 = tar
            elif tag == "1e-3":
                t3 = tar
            else:
                t2 = tar
        print(f"{desc:<20} {t4*100:>9.2f}% {t3*100:>9.2f}% {t2*100:>9.2f}%")


if __name__ == "__main__":
    main()