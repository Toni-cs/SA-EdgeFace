"""
LFW 数据一键准备：下载 + 解压 + 生成标准 pairs.txt。

用法:
    python scripts/prepare_lfw.py --output_dir datasets
    python scripts/prepare_lfw.py --output_dir datasets --skip_download  # 已有 lfw.tgz 时

产出:
    datasets/lfw/<name>/<name>_0001.jpg
    datasets/pairs.txt   (标准 10-fold 6000 对)
"""

import argparse
import os
import sys
import tarfile
import random
import urllib.request

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"


def download(url, dest):
    if os.path.exists(dest):
        print(f"[skip] {dest} 已存在")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def extract(tgz_path, output_dir):
    lfw_dir = os.path.join(output_dir, "lfw")
    if os.path.isdir(lfw_dir) and len(os.listdir(lfw_dir)) > 10:
        print(f"[skip] {lfw_dir} 已解压")
        return
    print(f"[extract] {tgz_path}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(output_dir)
    if os.path.isdir(os.path.join(output_dir, "lfw")) and not os.path.isdir(lfw_dir):
        pass
    for sub in ["lfw", "lfw_funneled"]:
        p = os.path.join(output_dir, sub)
        if os.path.isdir(p) and sub != "lfw":
            os.rename(p, lfw_dir)
    print(f"[ok] 解压完成: {lfw_dir}")


def generate_pairs(lfw_dir, output_path, num_folds=10, num_pairs_per_fold=300, seed=42):
    """
    若官方 pairs.txt 下载失败，从 lfw 目录生成标准格式 pairs.txt。
    格式: 第一行 "10 300"，之后每折 300 对（前 150 同人，后 150 不同人）。
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[skip] {output_path} 已存在且有效")
        return
    random.seed(seed)
    names = sorted([d for d in os.listdir(lfw_dir)
                    if os.path.isdir(os.path.join(lfw_dir, d))])
    name_imgs = {}
    for n in names:
        imgs = sorted([f for f in os.listdir(os.path.join(lfw_dir, n))
                       if f.endswith(".jpg")])
        if len(imgs) >= 2:
            name_imgs[n] = imgs
    names = list(name_imgs.keys())
    print(f"[pairs] {len(names)} identities with >=2 images, generating pairs...")

    lines = [f"{num_folds} {num_pairs_per_fold}"]
    for _ in range(num_folds):
        same = 0
        while same < num_pairs_per_fold // 2:
            n = random.choice(names)
            if len(name_imgs[n]) < 2:
                continue
            i1, i2 = random.sample(range(len(name_imgs[n])), 2)
            lines.append(f"{n} {i1+1} {i2+1}")
            same += 1
        diff = 0
        while diff < num_pairs_per_fold // 2:
            n1, n2 = random.sample(names, 2)
            i1 = random.randint(0, len(name_imgs[n1]) - 1)
            i2 = random.randint(0, len(name_imgs[n2]) - 1)
            lines.append(f"{n1} {i1+1} {n2} {i2+1}")
            diff += 1
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] pairs.txt 生成: {output_path} ({len(lines)-1} 对)")


def main():
    parser = argparse.ArgumentParser(description="Prepare LFW dataset")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tgz_path = os.path.join(args.output_dir, "lfw.tgz")
    pairs_path = os.path.join(args.output_dir, "pairs.txt")

    if not args.skip_download:
        download(LFW_URL, tgz_path)
        try:
            download(PAIRS_URL, pairs_path)
        except Exception as e:
            print(f"[warn] 官方 pairs.txt 下载失败: {e}，将自动生成")

    extract(tgz_path, args.output_dir)
    lfw_dir = os.path.join(args.output_dir, "lfw")
    if not os.path.isdir(lfw_dir):
        print(f"[error] {lfw_dir} 不存在，请检查解压")
        sys.exit(1)

    if not (os.path.exists(pairs_path) and os.path.getsize(pairs_path) > 1000):
        generate_pairs(lfw_dir, pairs_path)

    n_identities = len([d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))])
    print(f"\n[done] LFW 准备完成: {lfw_dir} ({n_identities} identities), pairs: {pairs_path}")


if __name__ == "__main__":
    main()