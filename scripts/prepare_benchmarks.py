"""T-R5 前处理：三基准原始图 → 112×112 对齐目录（与 align_lfw.py 同口径）。

口径（卡 :621「输入对齐至 112×112」+ 本项目 LFW 评测一致性）：
  - detect 模式（cfp_fp / agedb30 默认）：insightface buffalo_s 检测 + ArcFace 5 点模板
    norm_crop（与 align_lfw.py 同栈同参 det_size=640）；检不出 → 纯 resize 112 兜底落盘并
    如实计数（CFP 为官方 350×350 紧裁图，检出率预期高；兜底计数须进 DG-R0 披露）
  - resize 模式（tinyface 默认）：LR 人脸已预裁（约 20×16 量级），检测器在此尺度不可靠，
    评测惯例直接重采样到 112×112；可 --mode detect 强制检测
  - 输出目录**镜像源相对路径**（eval_generic_pairs 清单按相对路径解析，两件不改即插）：
      cfp_fp  : <src>/Data/Images/...            → <out>/Data/Images/...
      agedb30 : <src>/train/images/<shard>/...   → <out>/train/images/<shard>/...
      tinyface: <src>/<probe|gallery|lists>...   → <out>/<...>...
  - 幂等 resume：输出已存在即跳过；不删除、不覆盖任何已有文件

用法：
  python scripts/prepare_benchmarks.py --dataset cfp_fp
  python scripts/prepare_benchmarks.py --dataset agedb30
  python scripts/prepare_benchmarks.py --dataset tinyface [--mode detect]
  自定义路径：--src <源根> --out <输出根>
  汇总落盘（可选，拒绝覆盖）：--summary results/_t5_prepare_cfp_20261015.json
"""
import argparse
import datetime
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.utils import imread_cn, imwrite_cn  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULTS = {
    "cfp_fp":  {"src": "datasets/raw/cfp/cfp-dataset",
                "out": "datasets/cfp_aligned",   "mode": "detect"},
    "agedb30": {"src": "datasets/raw/agedb_hf",
                "out": "datasets/agedb30_aligned", "mode": "detect"},
    "tinyface": {"src": "datasets/raw/tinyface",
                 "out": "datasets/tinyface_112", "mode": "resize"},
}


def iter_images(src_root):
    for dirpath, _dirs, files in os.walk(src_root):
        for fn in sorted(files):
            if fn.lower().endswith(IMG_EXT):
                full = os.path.join(dirpath, fn)
                yield os.path.relpath(full, src_root), full


def build_detector():
    """与 align_lfw.py 同栈同参。"""
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align  # noqa: F401  （norm_crop 用）
    app = FaceAnalysis(name="buffalo_s", root="weights",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, face_align


def main():
    ap = argparse.ArgumentParser(description="T-R5 三基准 112×112 对齐前处理")
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--mode", choices=["detect", "resize"], default=None,
                    help="默认：cfp/agedb=detect，tinyface=resize")
    ap.add_argument("--summary", type=str, default=None, help="汇总 JSON（已存在拒写）")
    args = ap.parse_args()

    cfg = DEFAULTS[args.dataset]
    src = args.src or cfg["src"]
    out_root = args.out or cfg["out"]
    mode = args.mode or cfg["mode"]
    if not os.path.isdir(src):
        print(f"[ERROR] 源目录不存在: {src}")
        sys.exit(2)
    if args.summary and os.path.exists(args.summary):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {args.summary}")
        sys.exit(5)

    app = face_align = None
    if mode == "detect":
        app, face_align = build_detector()

    n_total = n_skip = n_det = n_resize_fallback = n_unreadable = 0
    t0 = time.time()
    for rel, full in iter_images(src):
        n_total += 1
        dst = os.path.join(out_root, rel)
        if os.path.exists(dst):
            n_skip += 1
            continue
        img = imread_cn(full)
        if img is None or img.size == 0:
            n_unreadable += 1
            continue
        if mode == "detect":
            faces = app.get(img)
            if len(faces) == 0:
                out = cv2.resize(img, (112, 112))
                n_resize_fallback += 1
            else:
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                out = face_align.norm_crop(img, face.kps)
                n_det += 1
        else:
            out = cv2.resize(img, (112, 112))
            n_resize_fallback += 1
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        imwrite_cn(dst, out)
        done = n_det + n_resize_fallback
        if done % 2000 == 0:
            print(f"  written={done} det={n_det} resize={n_resize_fallback} "
                  f"skip={n_skip} ({time.time() - t0:.0f}s)", flush=True)

    summary = {
        "schema": "t5_prepare_v1",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "mode": mode,
        "src": os.path.abspath(src),
        "out": os.path.abspath(out_root),
        "n_total": n_total,
        "n_written_detect": n_det,
        "n_written_resize": n_resize_fallback,
        "n_skipped_existing": n_skip,
        "n_unreadable": n_unreadable,
        "elapsed_s": round(time.time() - t0, 1),
        "script": "scripts/prepare_benchmarks.py",
    }
    print(f"[done] total={n_total} det={n_det} resize={n_resize_fallback} "
          f"skip={n_skip} unreadable={n_unreadable} ({summary['elapsed_s']}s) → {out_root}")
    if n_unreadable:
        print(f"[WARN] 不可读 {n_unreadable} 张（须在 eval_generic_pairs --manifest_only 体检中=0）")
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        if os.path.exists(args.summary):
            print(f"[ERROR] 目标文件已存在，拒绝覆盖: {args.summary}")
            sys.exit(5)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  summary → {args.summary}")


if __name__ == "__main__":
    main()
