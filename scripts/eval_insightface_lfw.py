"""
用 InsightFace buffalo_s 的 w600k_mbf.onnx 在 LFW (对齐后) 上评测。
作为教师模型 baseline 对比。
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.utils import imread_cn
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw

LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"
ONNX_PATH = "weights/models/buffalo_s/w600k_mbf.onnx"
OUTPUT = "results/insightface_lfw.json"


def main():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Providers: {session.get_providers()}")

    pairs = load_pairs(PAIRS_PATH)
    print(f"LFW pairs: {len(pairs)}")

    needed = set()
    for name1, idx1, name2, idx2, is_same in pairs:
        needed.add(get_image_path(LFW_DIR, name1, idx1))
        needed.add(get_image_path(LFW_DIR, name2, idx2))
    print(f"Unique images: {len(needed)}")

    cache = {}
    missing = 0
    for i, p in enumerate(sorted(needed)):
        if not os.path.exists(p):
            cache[p] = np.zeros(512, dtype=np.float32)
            missing += 1
            continue
        img = imread_cn(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
        t = np.transpose(img, (2, 0, 1))[None]
        feat = session.run([output_name], {input_name: t})[0].flatten()
        cache[p] = feat
        if (i + 1) % 2000 == 0:
            print(f"    embedded {i+1}/{len(needed)}")
    if missing > 0:
        print(f"  WARNING: {missing} images missing")

    features = []
    is_same_list = []
    for name1, idx1, name2, idx2, is_same in pairs:
        p1 = get_image_path(LFW_DIR, name1, idx1)
        p2 = get_image_path(LFW_DIR, name2, idx2)
        features.append(cache[p1])
        features.append(cache[p2])
        is_same_list.append(is_same)

    features = np.array(features)
    r = evaluate_lfw(features, pairs, is_same_list)
    print(f"\nInsightFace buffalo_s (w600k_mbf) on LFW:")
    print(f"  Acc={r['accuracy']*100:.2f}%  AUC={r['auc']:.4f}  EER={r['eer']*100:.2f}%")
    print(f"  TAR@1e-3={r.get('tar@0.001',0)*100:.2f}%  TAR@1e-4={r.get('tar@0.0001',0)*100:.2f}%")

    result = {
        "model": "InsightFace buffalo_s (w600k_mbf)",
        "accuracy": r["accuracy"],
        "auc": r["auc"],
        "eer": r["eer"],
        "tar@1e-3": r.get("tar@0.001", 0.0),
        "tar@1e-4": r.get("tar@0.0001", 0.0),
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()