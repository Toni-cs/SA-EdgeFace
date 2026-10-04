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
import hashlib
import os
import sys
import tarfile
import urllib.request

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
PAIRS_MD5 = "9f1ba174e4e1c508ff7cdf10ac338a7d"


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


def verify_pairs_md5(pairs_path, expected_md5):
    """校验 pairs.txt 的 MD5 哈希，确保是官方标准文件。"""
    if not os.path.exists(pairs_path):
        return False
    with open(pairs_path, "rb") as f:
        actual = hashlib.md5(f.read()).hexdigest()
    if actual == expected_md5:
        print(f"[ok] pairs.txt MD5 校验通过: {actual}")
        return True
    else:
        print(f"[error] pairs.txt MD5 不匹配!")
        print(f"  期望: {expected_md5}")
        print(f"  实际: {actual}")
        return False


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
            print(f"[error] 官方 pairs.txt 下载失败: {e}")
            print(f"[error] 请手动从 {PAIRS_URL} 下载到 {pairs_path}")
            sys.exit(1)

    extract(tgz_path, args.output_dir)
    lfw_dir = os.path.join(args.output_dir, "lfw")
    if not os.path.isdir(lfw_dir):
        print(f"[error] {lfw_dir} 不存在，请检查解压")
        sys.exit(1)

    if not verify_pairs_md5(pairs_path, PAIRS_MD5):
        print(f"[error] pairs.txt 不存在或 MD5 校验失败!")
        print(f"[error] 请手动从 {PAIRS_URL} 下载官方 pairs.txt 到 {pairs_path}")
        print(f"[error] 拒绝使用非官方 pairs 文件——评测可靠性要求标准 10-fold 协议")
        sys.exit(1)

    n_identities = len([d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))])
    print(f"\n[done] LFW 准备完成: {lfw_dir} ({n_identities} identities), pairs: {pairs_path}")


if __name__ == "__main__":
    main()