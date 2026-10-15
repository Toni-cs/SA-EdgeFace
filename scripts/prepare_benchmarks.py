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
import numpy as np

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


# ---- fiducial 对齐（CFP 官方 30 点标注；schema 2026-10-15 坐标级锁定，D-35）----
# 13-18 眼环（13外/14,15上/16内/17,18下）→ 眼心=centroid；22-25 鼻（22梁底+翼）
# → 鼻点=centroid；26/28 口角 → 口心=中点；槽位：正面=像左槽；profile 按鼻点.x
# 相对眼心.x 判向（鼻朝右=见其右侧=其右眼=正位照中居像左=L槽；解剖×三角双推导）。
TPL_EYE_L = (38.2946, 51.6963)     # ArcFace 112 模板（与 insightface norm_crop 同源）
TPL_EYE_R = (73.5318, 51.5014)
TPL_NOSE = (56.0252, 71.7366)
TPL_MOUTH_L = (41.5493, 92.3655)
TPL_MOUTH_R = (70.7299, 92.2041)
TPL_MOUTH_C = ((TPL_MOUTH_L[0] + TPL_MOUTH_R[0]) / 2.0,
               (TPL_MOUTH_L[1] + TPL_MOUTH_R[1]) / 2.0)


def load_fiducial(fid_path):
    """读 30 点 x,y 逐行；非 30 行/缺失 → None。"""
    if not os.path.exists(fid_path):
        return None
    pts = []
    with open(fid_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = line.split(",")
                pts.append((float(x), float(y)))
    if len(pts) != 30:
        return None
    return np.asarray(pts, dtype=np.float64)


def fiducial_features(pts, view):
    """→ (眼心, 鼻点, 口心, 槽位"L"/"R")。

    判向：frontal 恒 L 槽；profile 主判据 = 轮廓下颌端(点9) − 耳端(点1) 的
    x 差符号——俯仰下鼻点水平位移失稳而下颌-耳端保持（D-35-pre 跑前登记：
    与旧判 sign(鼻心.x−眼心.x) 全量 1974/2000=98.70% 一致，26 分歧中
    23 例 |dx|<10px 恰为旧判据掷硬币区）。
    """
    eye = pts[12:18].mean(axis=0)
    nose = pts[21:25].mean(axis=0)
    mouth = (pts[25] + pts[27]) / 2.0
    if view == "frontal":
        slot = "L"
    else:
        slot = "L" if pts[8, 0] > pts[0, 0] else "R"
    return eye, nose, mouth, slot


def umeyama_similarity(src, dst):
    """闭式相似变换（无反射，确定性）：dst ≈ s·R·src + t。src/dst: 3×2。"""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    cs = src - mu_s
    cd = dst - mu_d
    var_s = float((cs ** 2).sum()) / len(src)
    cov = cd.T @ cs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S)) / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    M = np.zeros((2, 3))
    M[:, :2] = s * R
    M[:, 2] = t
    return M


def align_fiducial(img, fid_path, view):
    """官方标注相似对齐 → 112×112；标注不可用 → None（调用方兜底）。

    frontal: 3 点（眼心/鼻点/口心）拟合（首轮即此，正面无恙不动）；
    profile: 2 点（眼心/口心）拟合——鼻点在偏航下前缩/侧移是 3 点拟合残差
    首凶（D-35-pre 机制判别：resid_nose 判别力 d=0.871，漏判 p90=11.6px vs
    正确 5.4px），剔出拟合；口心=两口角中点，绕垂直轴旋转保持矢状面投影
    （偏航不变量），与眼心共定尺度与转角。
    """
    pts = load_fiducial(fid_path)
    if pts is None:
        return None
    eye, nose, mouth, slot = fiducial_features(pts, view)
    eye_tpl = TPL_EYE_L if slot == "L" else TPL_EYE_R
    if view == "profile":
        src = np.array([eye, mouth])
        dst = np.array([eye_tpl, TPL_MOUTH_C])
    else:
        src = np.array([eye, nose, mouth])
        dst = np.array([eye_tpl, TPL_NOSE, TPL_MOUTH_C])
    M = umeyama_similarity(src, dst)
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description="T-R5 三基准 112×112 对齐前处理")
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--mode", choices=["detect", "resize", "fiducial"], default=None,
                    help="默认：cfp/agedb=detect，tinyface=resize；fiducial=CFP 官方 30 点标注对齐（须显式指定）")
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
    n_fid = n_fid_fallback = 0
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
        elif mode == "fiducial":
            norm = rel.replace("\\", "/")
            view = "frontal" if "/frontal/" in norm else "profile"
            fid = os.path.join(src, norm.replace("Data/Images", "Data/Fiducials"))
            fid = os.path.splitext(fid)[0] + ".txt"
            out = align_fiducial(img, fid, view)
            if out is None:
                out = cv2.resize(img, (112, 112))
                n_fid_fallback += 1
            else:
                n_fid += 1
        else:
            out = cv2.resize(img, (112, 112))
            n_resize_fallback += 1
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        imwrite_cn(dst, out)
        done = n_det + n_resize_fallback + n_fid
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
        "n_written_fiducial": n_fid,
        "n_fid_fallback": n_fid_fallback,
        "n_skipped_existing": n_skip,
        "n_unreadable": n_unreadable,
        "elapsed_s": round(time.time() - t0, 1),
        "script": "scripts/prepare_benchmarks.py",
    }
    print(f"[done] total={n_total} det={n_det} resize={n_resize_fallback} "
          f"fid={n_fid} fidfb={n_fid_fallback} "
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
