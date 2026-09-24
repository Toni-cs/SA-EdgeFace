"""
统一评测脚本：对训练好的 EdgeFace 模型在 LFW 和 QMUL-SurvFace 上评测。

用法:
    # 评测单个模型
    python scripts/eval_model.py --checkpoint weights/ablation/cfg3_final.pt --config x05_sab_full --lfw

    # 批量评测所有消融配置
    python scripts/eval_model.py --batch --save_dir weights/ablation --lfw

    # QMUL-SurvFace 评测
    python scripts/eval_model.py --checkpoint weights/ablation/cfg3_final.pt --config x05_sab_full --qmul --qmul_dir datasets/qmul_survface
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw, print_results
from qad_face.evaluation.qmul_eval import load_qmul_pairs, evaluate_qmul, evaluate_by_width, print_qmul_results


def load_model(checkpoint_path, config_name, device="cpu"):
    model = build_edgeface(DEFAULT_CONFIGS[config_name]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    return model


def extract_feature(model, img_path, device="cpu"):
    img = imread_cn(img_path)
    if img is None:
        return np.zeros(512, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.get_embedding(t).cpu().numpy().flatten()
    return feat


def eval_lfw(model, lfw_dir, pairs_path, device="cpu", model_name="Model"):
    pairs = load_pairs(pairs_path)
    print(f"  LFW pairs: {len(pairs)}")

    needed = set()
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(lfw_dir, name1, idx1))
        needed.add(get_image_path(lfw_dir, name2, idx2))

    cache = {}
    missing = 0
    for p in sorted(needed):
        if not os.path.exists(p):
            cache[p] = np.zeros(512, dtype=np.float32)
            missing += 1
            continue
        cache[p] = extract_feature(model, p, device)

    if missing > 0:
        print(f"  WARNING: {missing} images not found (using zero features)")

    features = []
    is_same_list = []
    for name1, idx1, name2, idx2, is_same in pairs:
        p1 = get_image_path(lfw_dir, name1, idx1)
        p2 = get_image_path(lfw_dir, name2, idx2)
        features.append(cache[p1])
        features.append(cache[p2])
        is_same_list.append(is_same)

    features = np.array(features)
    results = evaluate_lfw(features, pairs, is_same_list)
    print_results(results, model_name)
    return results


def eval_qmul(model, qmul_dir, device="cpu", model_name="Model"):
    pairs_path = os.path.join(qmul_dir, "pairs.txt")
    img_dir = os.path.join(qmul_dir, "images")
    pairs = load_qmul_pairs(pairs_path)
    print(f"  QMUL pairs: {len(pairs)}")

    needed = set()
    for img1, img2, is_same, w1, w2 in pairs:
        needed.add(os.path.join(img_dir, img1))
        needed.add(os.path.join(img_dir, img2))

    cache = {}
    for p in sorted(needed):
        cache[p] = extract_feature(model, p, device)

    features = []
    is_same_list = []
    widths = []
    for img1, img2, is_same, w1, w2 in pairs:
        p1 = os.path.join(img_dir, img1)
        p2 = os.path.join(img_dir, img2)
        features.append(cache[p1])
        features.append(cache[p2])
        is_same_list.append(is_same)
        widths.append(min(w1 or 999, w2 or 999))

    features = np.array(features)
    results = evaluate_qmul(features, pairs, is_same_list)

    by_width = None
    if any(w is not None and w < 999 for w in widths):
        sims = []
        for i in range(len(pairs)):
            f1, f2 = features[2*i], features[2*i+1]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            sims.append(sim)
        by_width = evaluate_by_width(sims, [1 if s else 0 for s in is_same_list], widths)

    print_qmul_results(results, by_width, model_name)
    return results


BATCH_CONFIGS = [
    ("cfg0", "x05_baseline",    "baseline"),
    ("cfg1", "x05_sab_channel", "+C1 channel"),
    ("cfg2", "x05_sab_spatial", "+C1 spatial"),
    ("cfg3", "x05_sab_full",    "+C1 full SAB"),
    ("cfg4", "x05_sab_full",    "+C1+C2 distill"),
    ("cfg5", "x05_sab_full",    "+C1+C2+C3 aug"),
]


def main():
    parser = argparse.ArgumentParser(description="模型评测")
    parser.add_argument("--checkpoint", help="单个模型 checkpoint 路径")
    parser.add_argument("--config", help="模型配置名 (如 x05_sab_full)")
    parser.add_argument("--batch", action="store_true", help="批量评测所有消融配置")
    parser.add_argument("--save_dir", default="weights/ablation", help="批量模式下 checkpoint 目录")
    parser.add_argument("--lfw", action="store_true", help="评测 LFW")
    parser.add_argument("--qmul", action="store_true", help="评测 QMUL-SurvFace")
    parser.add_argument("--lfw_dir", default="datasets/lfw")
    parser.add_argument("--lfw_pairs", default="datasets/pairs.txt")
    parser.add_argument("--qmul_dir", default="datasets/qmul_survface")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/eval_results.txt", help="结果保存路径")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")

    all_results = []

    if args.batch:
        for save_name, cfg_name, desc in BATCH_CONFIGS:
            ckpt_path = os.path.join(args.save_dir, f"{save_name}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {desc}: {ckpt_path} not found")
                continue
            print(f"\n{'='*60}")
            print(f"  Evaluating: {desc} ({save_name})")
            print(f"{'='*60}")
            model = load_model(ckpt_path, cfg_name, device)
            r = {"desc": desc}
            if args.lfw:
                r["lfw"] = eval_lfw(model, args.lfw_dir, args.lfw_pairs, device, desc)
            if args.qmul:
                r["qmul"] = eval_qmul(model, args.qmul_dir, device, desc)
            all_results.append(r)
    else:
        if not args.checkpoint or not args.config:
            print("ERROR: --checkpoint and --config required for single model eval")
            return
        model = load_model(args.checkpoint, args.config, device)
        r = {"desc": args.config}
        if args.lfw:
            r["lfw"] = eval_lfw(model, args.lfw_dir, args.lfw_pairs, device, args.config)
        if args.qmul:
            r["qmul"] = eval_qmul(model, args.qmul_dir, device, args.config)
        all_results.append(r)

    if args.output and all_results:
        with open(args.output, "w") as f:
            f.write("评测结果\n\n")
            for r in all_results:
                f.write(f"=== {r['desc']} ===\n")
                if "lfw" in r:
                    l = r["lfw"]
                    f.write(f"  LFW: Acc={l['accuracy']*100:.2f}% AUC={l['auc']:.4f} EER={l['eer']*100:.2f}%\n")
                if "qmul" in r:
                    q = r["qmul"]
                    f.write(f"  QMUL: Acc={q['accuracy']*100:.2f}% AUC={q['auc']:.4f} EER={q['eer']*100:.2f}%\n")
                f.write("\n")
        print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()