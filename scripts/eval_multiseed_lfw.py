"""
多 seed 评测：对 cfg0/cfg3/cfg4 各 3 seed 在 LFW 上做 10-fold CV 评测 + 配对 t 检验。

用法（在训练完成后运行）:
    python scripts/eval_multiseed_lfw.py

产出:
    results/multiseed_lfw.json   (每 config 每 seed 的指标 + t 检验结果)
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import cv2
from scipy import stats

sys.path.insert(0, ".")
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw

WEIGHTS_DIR = "weights/ablation_casia"
LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"

CONFIGS = {
    "cfg0": {"model_key": "x05_baseline", "desc": "baseline"},
    "cfg3": {"model_key": "x05_sab_full", "desc": "+full SAB"},
    "cfg4": {"model_key": "x05_sab_full", "desc": "+distill"},
}
SEEDS = [42, 123, 456]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features(model, image_paths, batch_size=64):
    """批量提取特征。"""
    cache = {}
    valid_paths = [p for p in image_paths if p not in cache]

    for i in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[i:i + batch_size]
        batch_tensors = []
        batch_valid = []
        for p in batch_paths:
            img = imread_cn(p)
            if img is None:
                print(f"[ERROR] 图片不可读，缺失即中止: {p}")
                sys.exit(3)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
            t = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            batch_tensors.append(t)
            batch_valid.append(p)

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            feats = model.get_embedding(batch).cpu().numpy()

        for j, p in enumerate(batch_valid):
            cache[p] = feats[j]

    return cache


def evaluate_config_seed(cfg_id, model_key, seed):
    """评测单个 config+seed。"""
    weight_path = os.path.join(WEIGHTS_DIR, f"{cfg_id}_seed{seed}_final.pt")
    if not os.path.exists(weight_path):
        print(f"  [skip] {weight_path} 不存在")
        return None

    print(f"  [load] {weight_path}")
    model = build_edgeface(DEFAULT_CONFIGS[model_key]).to(DEVICE)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()

    converged = True
    if "train_loss" in ckpt:
        final_loss = ckpt["train_loss"]
        converged = final_loss < 10.0
        print(f"  [check] final_loss={final_loss:.2f} converged={converged}")
    elif "epoch" in ckpt:
        print(f"  [info] epoch={ckpt['epoch']} (no train_loss in ckpt)")

    pairs = load_pairs(PAIRS_PATH)
    needed = set()
    for n1, i1, n2, i2, s in pairs:
        needed.add(get_image_path(LFW_DIR, n1, i1))
        needed.add(get_image_path(LFW_DIR, n2, i2))
    needed = sorted(needed)

    print(f"  [extract] {len(needed)} images...")
    cache = extract_features(model, needed)

    features = []
    is_same_list = []
    for n1, i1, n2, i2, s in pairs:
        p1 = get_image_path(LFW_DIR, n1, i1)
        p2 = get_image_path(LFW_DIR, n2, i2)
        if p1 not in cache:
            print(f"[ERROR] missing image {p1}")
            sys.exit(3)
        if p2 not in cache:
            print(f"[ERROR] missing image {p2}")
            sys.exit(3)
        features.append(cache[p1])
        features.append(cache[p2])
        is_same_list.append(s)

    result = evaluate_lfw(np.array(features), pairs, is_same_list,
                          save_pair_data=os.path.join("results", "pairs", f"{cfg_id}_seed{seed}"))
    result["converged"] = converged
    status = "OK" if converged else "NOT CONVERGED"
    print(f"  [done] Acc={result['accuracy']*100:.2f}% AUC={result['auc']:.4f} EER={result['eer']*100:.2f}% [{status}]")
    return result


def paired_t_test(metric_name, cfg_a, cfg_b, results):
    """配对 t 检验：比较两个 config 在多个 seed 上的指标。"""
    vals_a = []
    vals_b = []
    for seed in SEEDS:
        key_a = f"{cfg_a}_seed{seed}"
        key_b = f"{cfg_b}_seed{seed}"
        if key_a in results and key_b in results:
            if not results[key_a].get("converged", True) or not results[key_b].get("converged", True):
                continue
            vals_a.append(results[key_a][metric_name])
            vals_b.append(results[key_b][metric_name])

    if len(vals_a) < 2:
        return None

    vals_a = np.array(vals_a)
    vals_b = np.array(vals_b)
    diff = vals_b - vals_a
    t_stat, p_value = stats.ttest_rel(vals_b, vals_a)

    return {
        "metric": metric_name,
        "cfg_a": cfg_a,
        "cfg_b": cfg_b,
        "vals_a": vals_a.tolist(),
        "vals_b": vals_b.tolist(),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_seeds": len(vals_a),
        "significant": p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="输出 json（默认 results/multiseed_<时间戳>.json）")
    parser.add_argument("--commit", action="store_true", help="写 results/multiseed_lfw.json（仅当文件不存在）")
    args = parser.parse_args()
    if args.commit:
        out_path = "results/multiseed_lfw.json"
    elif args.out:
        out_path = args.out
    else:
        out_path = os.path.join("results", f"multiseed_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(out_path):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}")
        sys.exit(5)
    pairs = load_pairs(PAIRS_PATH)
    print(f"Pairs: {len(pairs)}, Device: {DEVICE}\n")

    all_results = {}
    for cfg_id, cfg_info in CONFIGS.items():
        print(f"\n=== {cfg_id} ({cfg_info['desc']}) ===")
        for seed in SEEDS:
            print(f"\n  seed={seed}:")
            result = evaluate_config_seed(cfg_id, cfg_info["model_key"], seed)
            if result:
                all_results[f"{cfg_id}_seed{seed}"] = result

    print("\n\n=== Paired t-tests ===")
    t_tests = []
    comparisons = [("cfg0", "cfg3"), ("cfg0", "cfg4"), ("cfg3", "cfg4")]
    metrics = ["accuracy", "auc", "eer"]
    for cfg_a, cfg_b in comparisons:
        for metric in metrics:
            t_result = paired_t_test(metric, cfg_a, cfg_b, all_results)
            if t_result:
                sig = "*" if t_result["significant"] else ""
                print(f"  {metric}: {cfg_a} vs {cfg_b}: "
                      f"mean_diff={t_result['mean_diff']:.6f} "
                      f"p={t_result['p_value']:.4f} {sig}")
                t_tests.append(t_result)

    output = {
        "results": all_results,
        "t_tests": t_tests,
        "configs": list(CONFIGS.keys()),
        "seeds": SEEDS,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()