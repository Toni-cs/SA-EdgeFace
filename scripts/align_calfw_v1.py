# -*- coding: utf-8 -*-
"""align_calfw_v1.py — CALFW 官方 landmarks 五点仿射对齐器（C2-M3b P4 派生）
路径：官方 landmarks（extracted_landmarks/*_5loc_attri.txt，5 点）→ SimilarityTransform
      → 112×112 ArcFace 模板；不经 SCRFD（绕开 buffalo_s det 缺陷）。
输入：datasets/raw/calfw/extracted/images/<name>.jpg + 对应 5loc txt
输出：datasets/calfw_aligned/<name>.jpg（112×112 JPEG q95，文件名与 pairs 引用一致）
"""
import os, sys, time, json
import numpy as np
import cv2
from skimage import transform as trans

ROOT = r"E:\下次比赛项目\face_recognition_research"
IMAGES = os.path.join(ROOT, "datasets", "raw", "calfw", "extracted", "images")
LM_DIR = os.path.join(ROOT, "datasets", "raw", "calfw", "extracted_landmarks")
OUT = os.path.join(ROOT, "datasets", "calfw_aligned")
LOG = os.path.join(ROOT, "results", "_align_calfw_v1_20261026.txt")
JPEG_Q = 95

TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


def imread_cn(p):
    b = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def imwrite_cn(p, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        raise RuntimeError("imencode fail " + p)
    with open(p, "wb") as f:
        f.write(buf.tobytes())


def load_lm(name):
    p = os.path.join(LM_DIR, name + "_5loc_attri.txt")
    if not os.path.exists(p):
        return None
    arr = np.loadtxt(p, dtype=np.float64)
    if arr.shape != (5, 2):
        return None
    return arr


def align(img, lm5):
    t = trans.SimilarityTransform()
    t.estimate(lm5, TEMPLATE)
    M = t.params[0:2, :]
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print("[task] images=%d lm_dir=%s out=%s" % (len(files), LM_DIR, OUT), flush=True)
    n_ok = n_skip = n_miss_lm = n_miss_img = 0
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write("# START %s total=%d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), len(files)))
        for i, fn in enumerate(files):
            name = os.path.splitext(fn)[0]
            dest = os.path.join(OUT, name + ".jpg")
            if os.path.exists(dest):
                n_skip += 1
                continue
            src = os.path.join(IMAGES, fn)
            img = imread_cn(src)
            if img is None:
                n_miss_img += 1
                print("[MISS-IMG] %s" % fn, flush=True)
                continue
            lm = load_lm(name)
            if lm is None:
                n_miss_lm += 1
                # 漏 landmarks：整图 resize 兜底（计数恒等，不丢弃）
                out = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            else:
                out = align(img, lm)
            imwrite_cn(dest, out)
            n_ok += 1
            if (i + 1) % 1000 == 0:
                dt = time.time() - t0
                print("[%d/%d] ok=%d skip=%d misslm=%d missimg=%d %.1f img/s"
                      % (i + 1, len(files), n_ok, n_skip, n_miss_lm, n_miss_img, (i + 1) / max(dt, 1e-9)),
                      flush=True)
        lf.write("# END %s ok=%d skip=%d misslm=%d missimg=%d\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), n_ok, n_skip, n_miss_lm, n_miss_img))
    n_out = sum(1 for x in os.listdir(OUT) if x.lower().endswith(".jpg"))
    print("[final] ok=%d skip=%d misslm=%d missimg=%d  dest_files=%d  elapsed=%.0fs"
          % (n_ok, n_skip, n_miss_lm, n_miss_img, n_out, time.time() - t0), flush=True)
    print("[selfcheck] dest=%d expect=%d %s" % (n_out, len(files), "PASS" if n_out == len(files) else "FAIL"), flush=True)


if __name__ == "__main__":
    main()