"""
LFW 基线评测入口脚本。

用法:
    python scripts/eval_lfw.py --model insightface_buffalo_s
    python scripts/eval_lfw.py --model onnx --onnx_path weights/edgeface.onnx
"""

import argparse
import os
import sys
import time
import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw, print_results
from qad_face.models.recognizer import InsightFaceRecognizer, ONNXRecognizer


def extract_lfw_features(recognizer, lfw_dir, pairs):
    """
    提取 LFW 所有人脸对的特征。

    返回:
        features: np.array, shape (2*N, feat_dim)
        is_same_list: list of bool
    """
    features = []
    is_same_list = []
    total = len(pairs)
    start = time.time()

    for i, (name1, idx1, name2, idx2, is_same) in enumerate(pairs):
        img_path1 = get_image_path(lfw_dir, name1, idx1)
        img_path2 = get_image_path(lfw_dir, name2, idx2)

        feat1 = recognizer.get_feature(img_path1)
        feat2 = recognizer.get_feature(img_path2)

        if feat1 is None or feat2 is None:
            # 检测失败时用零向量兜底
            feat1 = np.zeros(512, dtype=np.float32) if feat1 is None else feat1
            feat2 = np.zeros(512, dtype=np.float32) if feat2 is None else feat2

        features.append(feat1)
        features.append(feat2)
        is_same_list.append(is_same)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  Progress: {i+1}/{total} pairs, "
                  f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    return np.array(features), is_same_list


def main():
    parser = argparse.ArgumentParser(description="LFW Face Recognition Evaluation")
    parser.add_argument("--lfw_dir", type=str, default="datasets/lfw",
                        help="LFW 数据集目录")
    parser.add_argument("--pairs_path", type=str, default="datasets/pairs.txt",
                        help="LFW pairs.txt 路径")
    parser.add_argument("--model", type=str, default="insightface_buffalo_s",
                        choices=["insightface_buffalo_s", "insightface_buffalo_m",
                                 "insightface_buffalo_l", "onnx"],
                        help="识别模型类型")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="ONNX 模型路径（当 model=onnx 时使用）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径（npy 格式）")
    args = parser.parse_args()

    # 检查数据集
    if not os.path.exists(args.lfw_dir):
        print(f"[ERROR] LFW 目录不存在: {args.lfw_dir}")
        print("请先下载 LFW 数据集并解压到 datasets/lfw")
        return
    if not os.path.exists(args.pairs_path):
        print(f"[ERROR] pairs.txt 不存在: {args.pairs_path}")
        return

    # 加载 pairs
    print(f"[INFO] Loading pairs from {args.pairs_path}")
    pairs = load_pairs(args.pairs_path)
    print(f"[INFO] Loaded {len(pairs)} pairs")

    # 初始化识别器
    print(f"[INFO] Initializing recognizer: {args.model}")
    if args.model.startswith("insightface"):
        model_name = args.model.replace("insightface_", "")
        recognizer = InsightFaceRecognizer(model_name=model_name)
    elif args.model == "onnx":
        if args.onnx_path is None:
            print("[ERROR] 请指定 --onnx_path")
            return
        recognizer = ONNXRecognizer(onnx_path=args.onnx_path, name=args.onnx_path)
    else:
        print(f"[ERROR] 未知模型: {args.model}")
        return

    # 提取特征
    print(f"[INFO] Extracting features...")
    t0 = time.time()
    features, is_same_list = extract_lfw_features(recognizer, args.lfw_dir, pairs)
    extract_time = time.time() - t0
    print(f"[INFO] Feature extraction done in {extract_time:.1f}s "
          f"({extract_time/len(pairs)*1000:.1f}ms/pair)")

    # 保存特征
    if args.output:
        np.save(args.output, features)
        print(f"[INFO] Features saved to {args.output}")

    # 评测
    results = evaluate_lfw(features, pairs, is_same_list)
    results["extract_time_s"] = extract_time
    results["model"] = args.model
    print_results(results, model_name=args.model)

    # 保存结果到文件
    result_path = os.path.join("results", f"lfw_{args.model}.txt")
    os.makedirs("results", exist_ok=True)
    with open(result_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Results saved to {result_path}")


if __name__ == "__main__":
    main()
