"""单配置 L.5MFW 10-fold CV 评测。用法: python scripts/eval_lfw_single.py --cfg 0"""
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw

LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"
SAVE_DIR = "weights/ablation_casia"

CONFIGS = [
    ("cfg0", "x05_baseline",    "baseline"),
    ("cfg1", "x05_sab_channel", "+C1 channel"),
    ("cfg2", "x05_sab_spatial", "+C1 spatial"),
    ("cfg3", "x05_sab_full",    "+C1 full SAB"),
    ("cfg4", "x05_sab_full",    "+C1+C2 distill"),
    ("cfg5", "x05_sab_full",    "+C1+C2+C3 aug"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=int, default=0, help="config index 0-5")
    args = parser.parse_args()

    save_name, cfg_name, desc = CONFIGS[args.cfg]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Config: {desc} ({save_name})")

    pairs = load_pairs(PAIRS_PATH)
    needed = sorted(set())
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(LFW_DIR, name1, idx1))
        needed.add(get_image_path(LFW_DIR, name2, idx2))

    print("Preloading images...")
    imgs_cache = {}
    for p in needed:
        if os.path.exists(p):
            img = imread_cn(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                imgs_cache[p] = np.transpose(img, (2, 0, 1))
    print(f"  loaded {len(imgs_cache)}/{len(needed)}")

    ckpt_path = os.path.join(SAVE_DIR, f"{save_name}_final.pt")
    model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()

    valid_imgs = [imgs_cache[p] for p in needed if p in imgs_cache]
    valid_paths = [p for p in needed if p in imgs_cache]
    feats = np.zeros((len(valid_imgs), 512), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(valid_imgs), 256):
            e = min(s + 256, len(valid_imgs))
            batch = np.array(valid_imgs[s:e], dtype=np.float32)
            t = torch.from_numpy(batch).to(device)
            feats[s:e] = model.get_embedding(t).cpu().numpy()[:, :512]
    cache = dict(zip(valid_paths, feats))
    print("  features extracted")

    features = []
    is_same_list = []
    for name1, idx1, name2, idx2, is_same in pairs:
        p1 = get_image_path(LFW_DIR, name1, idx1)
        p2 = get_image_path(LFW_DIR, name2, idx2)
        features.append(cache.get(p1, np.zeros(512, dtype=np.float32)))
        features.append(cache.get(p2, np.zeros(512, dtype=np.float32)))
        is_same_list.append(is_same)

    r = evaluate_lfw(np.array(features), pairs, is_same_list)
    print(f"\n  Acc(10-fold CV) = {r['accuracy']*100:.2f}% +/- {r.get('accuracy_std',0)*100:.2f}%")
    print(f"  Acc(oracle)     = {r.get('accuracy_oracle',0)*100:.2f}%  [ref]")
    print(f"  AUC = {r['auc']:.4f}  EER = {r['eer']*100:.2f}%")
    print(f"  TAR@1e-3 = {r.get('tar@0.001',0)*100:.2f}%  TAR@1e-4 = {r.get('tar@0.0001',0)*100:.2f}%")

    out = f"results/lfw_10fold_cfg{args.cfg}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": save_name, "desc": desc, **r}, f, indent=2, ensure_ascii=False)
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()