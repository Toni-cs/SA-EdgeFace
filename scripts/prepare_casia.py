"""
CASIA-WebFace 数据一键准备：探测格式 → 解压/解析 → 转 identity folders → 生成 train_subset 与 train_mini。

支持的输入格式（放入 datasets/raw/）:
  1. insightface 打包 .bin（train.bin / casia-webface.bin 等，aligned 112x112 标准分发格式）
  2. zip / tar.gz 压缩包（内部为图片目录）
  3. 已解压的图片目录（identity folders 或其父目录）

用法:
    python scripts/prepare_casia.py --raw_dir datasets/raw --output_dir datasets
    python scripts/prepare_casia.py --raw_dir datasets/raw --mini_identities 500
"""

import argparse
import os
import sys
import glob
import random
import zipfile
import tarfile
import shutil
import numpy as np
import cv2


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_input(raw_dir):
    """探测 raw_dir 下的输入：优先 .bin，其次压缩包，其次目录。"""
    bins = sorted(glob.glob(os.path.join(raw_dir, "*.bin")) +
                  glob.glob(os.path.join(raw_dir, "*", "*.bin")))
    if bins:
        return "bin", bins[0]
    archives = sorted(glob.glob(os.path.join(raw_dir, "*.zip")) +
                      glob.glob(os.path.join(raw_dir, "*.tgz")) +
                      glob.glob(os.path.join(raw_dir, "*.tar.gz")))
    if archives:
        return "archive", archives[0]
    for root, dirs, files in os.walk(raw_dir):
        n_img = sum(1 for f in files if f.lower().endswith(IMG_EXTS))
        if n_img > 100:
            return "dir", root
    return None, None


def read_bin_file(bin_path):
    """
    读 insightface 打包 bin: 每条记录 flag(u64) + label(f64) + length(u64) + jpeg bytes。
    flag==0 为边界记录，跳过。
    """
    labels, jpegs = [], []
    with open(bin_path, "rb") as f:
        while True:
            head = f.read(24)
            if len(head) < 24:
                break
            flag = np.frombuffer(head[:8], dtype=np.uint64)[0]
            label = np.frombuffer(head[8:16], dtype=np.float64)[0]
            length = int(np.frombuffer(head[16:24], dtype=np.uint64)[0])
            if flag == 0:
                continue
            data = f.read(length)
            if len(data) < length:
                break
            labels.append(int(label))
            jpegs.append(data)
    return labels, jpegs


def extract_archive(archive_path, dest):
    print(f"[extract] {archive_path}")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive_path) as t:
            t.extractall(dest)


def normalize_from_dir(src_dir, out_raw):
    """把嵌套图片目录规范化为 out_raw/<identity>/<n>.jpg。"""
    id_count = 0
    img_count = 0
    for root, dirs, files in os.walk(src_dir):
        imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        rel = os.path.relpath(root, src_dir)
        id_name = rel.replace(os.sep, "_") if rel != "." else "identities"
        out_id = os.path.join(out_raw, id_name)
        os.makedirs(out_id, exist_ok=True)
        for f in imgs:
            shutil.copy2(os.path.join(root, f), out_id)
            img_count += 1
        id_count += 1
    return id_count, img_count


def write_from_bin(labels, jpegs, out_raw):
    for i, (lb, jpg) in enumerate(zip(labels, jpegs)):
        out_id = os.path.join(out_raw, str(lb))
        os.makedirs(out_id, exist_ok=True)
        with open(os.path.join(out_id, f"{i:07d}.jpg"), "wb") as f:
            f.write(jpg)
    n_ids = len(set(labels))
    return n_ids, len(labels)


def filter_and_split(raw_dir, out_dir, mini_identities, min_samples=2):
    """过滤每身份样本数，生成 train_subset（全量）与 train_mini（调试子集）。"""
    identities = [d for d in os.listdir(raw_dir)
                  if os.path.isdir(os.path.join(raw_dir, d))]
    valid = {}
    for d in identities:
        imgs = [f for f in os.listdir(os.path.join(raw_dir, d))
                if f.lower().endswith(IMG_EXTS)]
        if len(imgs) >= min_samples:
            valid[d] = imgs
    print(f"[filter] {len(identities)} 身份 -> {len(valid)} 有效 (>= {min_samples} 张)")

    subset_dir = os.path.join(out_dir, "train_subset")
    mini_dir = os.path.join(out_dir, "train_mini")
    for d in (subset_dir, mini_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    total_imgs = 0
    for d, imgs in valid.items():
        os.makedirs(os.path.join(subset_dir, d), exist_ok=True)
        for f in imgs:
            os.link(os.path.join(raw_dir, d, f), os.path.join(subset_dir, d, f)) \
                if hasattr(os, "link") else shutil.copy2(
                    os.path.join(raw_dir, d, f), os.path.join(subset_dir, d, f))
        total_imgs += len(imgs)

    mini_ids = random.sample(list(valid.keys()), min(mini_identities, len(valid)))
    mini_imgs = 0
    for d in mini_ids:
        os.makedirs(os.path.join(mini_dir, d), exist_ok=True)
        for f in valid[d]:
            src = os.path.join(subset_dir, d, f)
            dst = os.path.join(mini_dir, d, f)
            if hasattr(os, "link"):
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        mini_imgs += len(valid[d])

    print(f"[done] train_subset: {len(valid)} 身份 / {total_imgs} 张")
    print(f"[done] train_mini:   {len(mini_ids)} 身份 / {mini_imgs} 张 (调试用)")


def main():
    parser = argparse.ArgumentParser(description="Prepare CASIA-WebFace")
    parser.add_argument("--raw_dir", type=str, default="datasets/raw")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--mini_identities", type=int, default=500)
    parser.add_argument("--min_samples", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    kind, path = find_input(args.raw_dir)
    if kind is None:
        print(f"[error] {args.raw_dir} 下未找到 .bin / 压缩包 / 图片目录")
        print("请把下载的 CASIA-WebFace 文件放入该目录后重跑")
        sys.exit(1)
    print(f"[input] kind={kind} path={path}")

    raw_out = os.path.join(args.output_dir, "casia_raw")
    if os.path.isdir(raw_out):
        shutil.rmtree(raw_out)
    os.makedirs(raw_out)

    if kind == "bin":
        print(f"[parse] 读取 insightface bin ...")
        labels, jpegs = read_bin_file(path)
        print(f"[parse] {len(labels)} 张, {len(set(labels))} 身份")
        n_ids, n_imgs = write_from_bin(labels, jpegs, raw_out)
    elif kind == "archive":
        extract_archive(path, os.path.join(args.output_dir, "_extract_tmp"))
        n_ids, n_imgs = normalize_from_dir(
            os.path.join(args.output_dir, "_extract_tmp"), raw_out)
        shutil.rmtree(os.path.join(args.output_dir, "_extract_tmp"), ignore_errors=True)
    else:
        n_ids, n_imgs = normalize_from_dir(path, raw_out)
    print(f"[normalize] {n_ids} 身份 / {n_imgs} 张 -> {raw_out}")

    filter_and_split(raw_out, args.output_dir, args.mini_identities, args.min_samples)


if __name__ == "__main__":
    main()