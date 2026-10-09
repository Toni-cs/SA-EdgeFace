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
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint 路径（默认 weights/ablation_casia/<save_name>_final.pt）")
    parser.add_argument("--out", type=str, default=None, help="输出 json（默认 results/lfw_10fold_cfgN.json；已存在则拒绝覆盖）")
    parser.add_argument("--arch", choices=["edgeface", "mobilefacenet"], default=None, help="模型架构；--ckpt 指向 mbf 臂时必须显式指定")
    args = parser.parse_args()
    base = os.path.basename(args.ckpt or "").lower()
    if (("mbf" in base) or ("mobilefacenet" in base)) and args.arch != "mobilefacenet":
        print("[ERROR] --ckpt 指向 MobileFaceNet 臂但未显式 --arch mobilefacenet（拒绝静默错配）")
        sys.exit(6)
    if args.arch is None:
        args.arch = "edgeface"
    if args.arch == "mobilefacenet":
        print("[ERROR] 仓库内无 MobileFaceNet 实现，无法构建（组E 前置缺口，登记 CONFLICTS.md）")
        sys.exit(6)

    save_name, cfg_name, desc = CONFIGS[args.cfg]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Config: {desc} ({save_name})")

    pairs = load_pairs(PAIRS_PATH)
    needed = set()
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(LFW_DIR, name1, idx1))
        needed.add(get_image_path(LFW_DIR, name2, idx2))
    needed = sorted(needed)
    assert len(needed) > 0, "no LFW pairs loaded"

    print("Preloading images...")
    imgs_cache = {}
    for p in needed:
        if os.path.exists(p):
            img = imread_cn(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                imgs_cache[p] = np.transpose(img, (2, 0, 1))
    if len(imgs_cache) < len(needed):
        missing = [p for p in needed if p not in imgs_cache]
        print(f"[ERROR] {len(missing)} LFW images unreadable (e.g. {missing[0]})")
        sys.exit(1)
    print(f"  loaded {len(imgs_cache)}/{len(needed)}")

    ckpt_path = args.ckpt if args.ckpt else os.path.join(SAVE_DIR, f"{save_name}_final.pt")
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
    print(f"\n  Acc(CV) = {r['accuracy']*100:.2f}% +/- {r.get('accuracy_std',0)*100:.2f}%")
    print(f"  Acc(oracle)     = {r.get('accuracy_oracle',0)*100:.2f}%  [ref]")
    print(f"  AUC = {r['auc']:.4f}  EER = {r['eer']*100:.2f}%")
    print(f"  TAR@1e-3 = {r.get('tar@0.001',0)*100:.2f}%  TAR@1e-4 = {r.get('tar@0.0001',0)*100:.2f}%")

    out = args.out if args.out else f"results/lfw_10fold_cfg{args.cfg}.json"
    if os.path.exists(out):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out}（请传 --out <新路径>）")
        sys.exit(5)
    payload = {"config": save_name, "desc": desc, "ckpt": ckpt_path,
               "acc_cv": r["accuracy"], "gate_pass": bool(r["accuracy"] >= 0.92), **r}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()