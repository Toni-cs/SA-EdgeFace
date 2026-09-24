"""
教师模型一键准备：提前下载 insightface buffalo_s/l 到 weights/models/，
训练时无需等待在线下载。

用法:
    python scripts/prepare_teacher.py --model buffalo_s
    python scripts/prepare_teacher.py --model buffalo_l
"""

import argparse
import os
import sys
import zipfile
import urllib.request

MODELS = {
    "buffalo_s": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
    "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "buffalo_m": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
}


def main():
    parser = argparse.ArgumentParser(description="Prepare insightface teacher model")
    parser.add_argument("--model", type=str, default="buffalo_s", choices=list(MODELS.keys()))
    parser.add_argument("--root", type=str, default="weights")
    args = parser.parse_args()

    model_dir = os.path.join(args.root, "models", args.model)
    if os.path.isdir(model_dir) and any(f.endswith(".onnx") for f in os.listdir(model_dir)):
        print(f"[skip] {model_dir} 已就绪")
        return

    os.makedirs(model_dir, exist_ok=True)
    zip_path = os.path.join(args.root, f"{args.model}.zip")
    url = MODELS[args.model]
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[ok] {os.path.getsize(zip_path)/1e6:.1f} MB")

    print(f"[extract] -> {model_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(model_dir)
    os.remove(zip_path)

    onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    print(f"[done] {args.model} 就绪: {onnx_files}")


if __name__ == "__main__":
    main()