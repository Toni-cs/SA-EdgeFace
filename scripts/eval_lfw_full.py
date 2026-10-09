"""
对全量 CASIA 训练的 6 配置在 LFW (对齐后) 上评测，保存全部指标含 TAR@FAR。
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw

LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"
SAVE_DIR = "weights/ablation_casia"

BATCH_CONFIGS = [
    ("cfg0", "x05_baseline",    "baseline"),
    ("cfg1", "x05_sab_channel", "+C1 channel"),
    ("cfg2", "x05_sab_spatial", "+C1 spatial"),
    ("cfg3", "x05_sab_full",    "+C1 full SAB"),
    ("cfg4", "x05_sab_full",    "+C1+C2 distill"),
    ("cfg5", "x05_sab_full",    "+C1+C2+C3 aug"),
]


def load_model(ckpt_path, cfg_name, device):
    model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "student" not in ckpt:
        raise ValueError(f"[ERROR] {ckpt_path} 缺少 student 权重")
    if "epoch" in ckpt:
        print(f"    [ckpt] epoch={ckpt['epoch']}")
    if ckpt.get("epoch", 0) < 25:
        print("[ERROR] " + str(ckpt_path) + " epoch<25, 未完成训练，拒绝评估")
        sys.exit(4)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    return model


def extract_feature(model, img_path, device):
    img = imread_cn(img_path)
    if img is None:
        print(f"[ERROR] 图片不可读，缺失即中止: {img_path}")
        sys.exit(3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.get_embedding(t).cpu().numpy().flatten()
    return feat


def preload_images(paths):
    imgs = {}
    for p in paths:
        if os.path.exists(p):
            img = imread_cn(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                imgs[p] = np.transpose(img, (2, 0, 1))
    if len(imgs) < len(paths):
        missing = [p for p in paths if p not in imgs]
        raise ValueError(f"[ERROR] {len(missing)} LFW 图片不可读 (e.g. {missing[0]})，评测中止")
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
            print(f"[ERROR] 预载缺失，缺失即中止: {p}")
            sys.exit(3)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="输出 json（默认 results/lfw_full_<时间戳>.json）")
    parser.add_argument("--commit", action="store_true", help="写 results/casia_lfw_full.json（仅当文件不存在）")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = load_pairs(PAIRS_PATH)
    print(f"LFW pairs: {len(pairs)}")

    needed = set()
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(LFW_DIR, name1, idx1))
        needed.add(get_image_path(LFW_DIR, name2, idx2))
    needed = sorted(needed)
    print(f"Unique images to embed: {len(needed)}")

    print("Preloading images...")
    imgs_cache = preload_images(needed)
    print(f"  loaded {len(imgs_cache)} images, missing: {len(needed) - len(imgs_cache)}")

    all_results = []
    missing = []
    for save_name, cfg_name, desc in BATCH_CONFIGS:
        ckpt_path = os.path.join(SAVE_DIR, f"{save_name}_final.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP {desc}: {ckpt_path} not found")
            missing.append(save_name)
            continue
        print(f"\n{'='*60}")
        print(f"  {desc} ({save_name})")
        print(f"{'='*60}")

        try:
            model = load_model(ckpt_path, cfg_name, device)
        except ValueError as e:
            print(e)
            missing.append(save_name)
            continue
        cache = batch_extract(model, needed, device, imgs_cache)
        print(f"    features extracted")

        features = []
        is_same_list = []
        for name1, idx1, name2, idx2, is_same in pairs:
            p1 = get_image_path(LFW_DIR, name1, idx1)
            p2 = get_image_path(LFW_DIR, name2, idx2)
            features.append(cache[p1])
            features.append(cache[p2])
            is_same_list.append(is_same)

        features = np.array(features)
        pair_dir = os.path.join("results", "pairs", save_name)
        r = evaluate_lfw(features, pairs, is_same_list, save_pair_data=pair_dir)
        print(f"  Acc(CV)={r['accuracy']*100:.2f}% +/- {r.get('accuracy_std',0)*100:.2f}%  Acc(oracle)={r.get('accuracy_oracle',0)*100:.2f}%")
        print(f"  AUC={r['auc']:.4f}  EER={r['eer']*100:.2f}%")
        print(f"  TAR@1e-3={r.get('tar@0.001',0)*100:.2f}%  TAR@1e-4={r.get('tar@0.0001',0)*100:.2f}%")

        all_results.append({
            "config": save_name,
            "desc": desc,
            "accuracy": r["accuracy"],
            "accuracy_std": r.get("accuracy_std", 0.0),
            "accuracy_oracle": r.get("accuracy_oracle", 0.0),
            "auc": r["auc"],
            "eer": r["eer"],
            "tar@1e-3": r.get("tar@0.001", 0.0),
            "tar@1e-4": r.get("tar@0.0001", 0.0),
            "best_threshold": r["best_threshold"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if missing:
        print(f"\n[ERROR] 缺失 {len(missing)}/{len(BATCH_CONFIGS)} 配置: {missing}")
        print("[ERROR] 拒绝覆写输出文件")
        sys.exit(3)

    if args.commit:
        out_path = "results/casia_lfw_full.json"
    elif args.out:
        out_path = args.out
    else:
        out_path = os.path.join("results", f"lfw_full_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(out_path):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}")
        sys.exit(5)   # S8: 目标文件冲突
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nresults saved: {out_path}")

    print(f"\n{'='*80}")
    print(f"{'Config':<25} {'Acc(CV)%':>8} {'+/-':>5} {'Acc(orc)%':>9} {'AUC':>7} {'EER%':>7} {'TAR@1e-3':>9} {'TAR@1e-4':>9}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['desc']:<25} {r['accuracy']*100:>8.2f} {r.get('accuracy_std',0)*100:>4.2f} {r.get('accuracy_oracle',0)*100:>8.2f} {r['auc']:>7.4f} {r['eer']*100:>7.2f} {r['tar@1e-3']*100:>8.2f}% {r['tar@1e-4']*100:>8.2f}%")


if __name__ == "__main__":
    main()