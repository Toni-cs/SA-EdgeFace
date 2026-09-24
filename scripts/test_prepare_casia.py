import os
import shutil
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

raw = "datasets/raw"
shutil.rmtree(raw, ignore_errors=True)
os.makedirs(raw, exist_ok=True)

with open("datasets/raw/train.bin", "wb") as f:
    n = 0
    for label in range(3):
        for _ in range(4):
            img = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            data = buf.tobytes()
            f.write(np.uint64(0xDEFABCEC).tobytes())
            f.write(np.float64(label).tobytes())
            f.write(np.uint64(len(data)).tobytes())
            f.write(data)
            n += 1
print(f"wrote {n} records")

from scripts.prepare_casia import read_bin_file, normalize_from_dir, filter_and_split

labels, jpegs = read_bin_file("datasets/raw/train.bin")
assert len(labels) == 12 and len(set(labels)) == 3, f"bin parse fail: {len(labels)}"
print(f"[OK] bin 解析: {len(labels)} 条, {len(set(labels))} 身份")

raw_out = "datasets/casia_raw"
shutil.rmtree(raw_out, ignore_errors=True)
os.makedirs(raw_out)
write_ok = normalize_from_dir if False else None
from scripts.prepare_casia import write_from_bin
n_ids, n_imgs = write_from_bin(labels, jpegs, raw_out)
assert n_ids == 3 and n_imgs == 12
print(f"[OK] identity folders: {n_ids} 身份 / {n_imgs} 张")

filter_and_split(raw_out, "datasets", mini_identities=2, min_samples=2)
subset = os.path.join("datasets", "train_subset")
mini = os.path.join("datasets", "train_mini")
n_subset = len(os.listdir(subset))
n_mini = len(os.listdir(mini))
assert n_subset == 3 and n_mini == 2, f"split fail: subset={n_subset} mini={n_mini}"
sample = os.listdir(os.path.join(subset, "0"))
assert len(sample) == 4, f"img copy fail: {sample}"
print(f"[OK] train_subset: {n_subset} 身份, train_mini: {n_mini} 身份")

shutil.rmtree(raw_out, ignore_errors=True)
shutil.rmtree(subset, ignore_errors=True)
shutil.rmtree(mini, ignore_errors=True)
shutil.rmtree(raw, ignore_errors=True)
print("[ALL PASSED] prepare_casia 全流程验证通过（bin解析/目录化/过滤/子集划分）")