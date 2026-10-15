"""T-R5：多基准评估管线扩展（prereg_v2_matrix pin 1 件）。

卡原文逐字口径（_wp2_extract:616-626）：
  1. CFP-FP：正面-侧面协议，10-fold，同 LFW 的 pairwise verification 逻辑（目录结构自适应）
  2. AgeDB-30：年龄间隔 30 的 pairwise 协议，10-fold
  3. TinyFace：probe-gallery Rank-1 协议（非 pairwise）；输入对齐至 112×112
  4. 三个新基准的评估结果统一写入与现有 LFW 相同格式的 JSON 报告
     {model_path, dataset, metric, value, config_hash}
  5. 门2「综合门」条款已被裁决二作废：门逻辑集中 gate_check.py（§13 绝对健康门），本件只出数
  6. 全部协议严格遵循各基准官方定义，禁止简化

人工必查点（卡 :604-609 / :705）：
  - TinyFace 的 Rank-1 与 pairwise 协议不同，禁止把所有基准统一成 pairwise
    （本件以 KIND_OF_DATASET + assert_kind_matches 双保险，单测 T5-* 专职锁死）
  - 用已发表模型跑一遍校准数值 → --calib_onnx（如 weights/models/buffalo_s/w600k_mbf.onnx）

协议来源（全部官方件，本机落盘）：
  CFP-FP    datasets/raw/cfp/cfp-dataset/Protocol/{Pair_list_*.txt,Split/<FP|FF>/<01..10>/{same,diff}.txt}
  AgeDB-30  --pairs_txt 指向官方 6000 对行式文件（<nameL> <nameR> <label>，10 折每折 600）
  TinyFace  --aligned_root 下 probe/gallery 目录树，或 probe.txt/gallery.txt 列表（行式 <relpath> <identity>）

用法：
  # 清单构建 + 数据体检（无 GPU 可跑，供冻结前验证）
  python scripts/eval_generic_pairs.py --dataset cfp_fp --manifest_only
  # 正式评测（EdgeFace 臂）
  python scripts/eval_generic_pairs.py --dataset cfp_fp --ckpt weights/xxx.pt \
      --backbone pconv --cfg cfg4 --protocol FP --out results/eval_cfpfp_xxx.json
  python scripts/eval_generic_pairs.py --dataset agedb30 --ckpt ... --out results/...json
  python scripts/eval_generic_pairs.py --dataset tinyface --ckpt ... --out results/...json
  # 已发表模型校准（DG-R0a）
  python scripts/eval_generic_pairs.py --dataset cfp_fp --calib_onnx weights/models/buffalo_s/w600k_mbf.onnx \
      --out results/_t5_calibration_cfpfp.json

红线：--out 已存在拒绝覆盖（exit 5）；任何图像缺失/不可读立即退出（不许静默丢对）。
"""
import os
import sys
import json
import csv
import argparse
import hashlib
import datetime

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_curve, auc                      # noqa: E402
from qad_face.utils import imread_cn                            # noqa: E402
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface  # noqa: E402

DATASET_CHOICES = ("cfp_fp", "agedb30", "tinyface")
DATASET_CANON = {"cfp_fp": "CFP-FP", "agedb30": "AgeDB-30", "tinyface": "TinyFace"}
# 协议种类映射（卡 :620 :705：TinyFace 只许 probe-gallery Rank-1，禁止 pairwise 化）
KIND_OF_DATASET = {"cfp_fp": "pairwise", "agedb30": "pairwise", "tinyface": "rank1"}
METRIC_OF_KIND = {"pairwise": "acc_cv10", "rank1": "rank1"}

DEFAULT_ALIGNED = {
    "cfp_fp": "datasets/cfp_aligned",
    "agedb30": "datasets/agedb30_aligned",
    "tinyface": "datasets/tinyface_112",
}
DEFAULT_PROTOCOL_ROOT = "datasets/raw/cfp/cfp-dataset"
DEFAULT_AGEDB_PAIRS = "datasets/raw/agedb30/agedb_30.txt"
MANIFEST_DIR = "datasets/eval_manifests"
PREPROCESS_DESC = "bgr_rgb_resize112_div127.5_-1"   # 与 eval_lfw_single 逐位同口径


# ============================ 工具 ============================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_config_hash(cfg):
    """config_hash = sha256(排序 JSON(评测配置))，与模型无关（同协议不同模型可比）。"""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_payload(model_path, dataset, metric, value, config_hash, **extra):
    """统一 JSON 报告（卡 :622-623 必含五键，顺序固定）。"""
    p = {
        "model_path": model_path,
        "dataset": dataset,
        "metric": metric,
        "value": value,
        "config_hash": config_hash,
    }
    p.update(extra)
    return p


# ============================ 协议层（纯函数，单测直测） ============================

def _cfp_locate(root):
    """接受 cfp-dataset/ 根或其 Protocol/ 本身。"""
    for cand in (root, os.path.join(root, "Protocol")):
        if os.path.isdir(os.path.join(cand, "Split")):
            return cand
    raise FileNotFoundError(f"CFP 官方协议未找到（Split/ 缺失）: {root}")


def _read_pair_list(path):
    """行式 '<num> <relpath>' → dict[int,str]；'../' 前缀剥掉。"""
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{ln} 行式应为 '<num> <relpath>'，实为 {line!r}")
            m[int(parts[0])] = parts[1].lstrip("./") if parts[1].startswith("../") else parts[1]
    return m


def load_cfp_protocol(root, protocol):
    """CFP 官方协议解析。

    官方编号：Pair_list_F(1..5000 frontal)、Pair_list_P(1..2000 profile)；
    Split/<protocol>/<01..10>/{same,diff}.txt 行式 '<a>,<b>'。
      FP：a → F 表，b → P 表（正面-侧面，卡 :617）
      FF：a、b → F 表（正面-正面，对照协议）
    返回 rows: [{'fold': 0..9, 'label': 0/1, 'p1': rel, 'p2': rel}, ...]，按折聚集。
    """
    if protocol not in ("FP", "FF"):
        raise ValueError(f"CFP 协议只许 FP/FF，收到 {protocol!r}")
    pdir = _cfp_locate(root)
    F = _read_pair_list(os.path.join(pdir, "Pair_list_F.txt"))
    P = _read_pair_list(os.path.join(pdir, "Pair_list_P.txt")) if protocol == "FP" else None
    split_dir = os.path.join(pdir, "Split", protocol)
    fold_dirs = sorted(d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d)))
    if not fold_dirs:
        raise FileNotFoundError(f"CFP 折目录缺失: {split_dir}")
    rows = []
    for fi, fd in enumerate(fold_dirs):
        for label, fn in ((1, "same.txt"), (0, "diff.txt")):
            fp = os.path.join(split_dir, fd, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"CFP 官方折文件缺失: {fp}")
            with open(fp, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    parts = line.replace(",", " ").split()
                    if len(parts) != 2:
                        raise ValueError(f"{fp}:{ln} 行式应为 '<a>,<b>'，实为 {line!r}")
                    a, b = int(parts[0]), int(parts[1])
                    if a not in F or (protocol == "FF" and b not in F) or (protocol == "FP" and b not in P):
                        raise KeyError(f"{fp}:{ln} 编号越界: {line!r}")
                    rows.append({
                        "fold": fi,
                        "label": label,
                        "p1": F[a],
                        "p2": F[b] if protocol == "FF" else P[b],
                    })
    return rows


def load_agedb_pairs(pairs_txt, n_folds=10):
    """AgeDB-30 官方 6000 对。行式 '<nameL> <nameR> <label>'（label ∈ {0,1,-1}，-1→0）。
    官方折号：fold = i // (n_lines // n_folds)（文件序即折序，每折 600）。
    """
    with open(pairs_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) == 0:
        raise ValueError(f"空对文件: {pairs_txt}")
    if len(lines) % n_folds != 0:
        raise ValueError(f"{pairs_txt}: {len(lines)} 行不能被 {n_folds} 折整除")
    per = len(lines) // n_folds
    rows = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{pairs_txt}:{i + 1} 行式应为 '<nameL> <nameR> <label>'，实为 {line!r}")
        raw = parts[2]
        if raw in ("0", "1"):
            label = int(raw)
        elif raw == "-1":
            label = 0
        else:
            raise ValueError(f"{pairs_txt}:{i + 1} label 只许 0/1/-1，实为 {raw!r}")
        rows.append({"fold": i // per, "label": label, "p1": parts[0], "p2": parts[1],
                     "_n1": parts[0], "_n2": parts[1]})
    return rows


def resolve_agedb_names(rows, img_root):
    """把对文件里的名字解析为 img_root 下真实相对路径：
    ① 直接拼接存在 → 用之；② 否则按文件名（basename）全库索引唯一命中 → 用之；
    ③ 缺失/重名歧义 → 硬错（禁止静默丢对）。"""
    idx = {}
    for dirpath, _dirs, files in os.walk(img_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                if fn in idx:
                    idx[fn] = None  # 标记歧义
                else:
                    idx[fn] = os.path.relpath(os.path.join(dirpath, fn), img_root)
    missing, ambiguous = [], []
    for r in rows:
        for key in ("p1", "p2"):
            name = r[key]
            direct = os.path.join(img_root, name)
            if os.path.exists(direct):
                r[key] = os.path.relpath(direct, img_root)
                continue
            base = os.path.basename(name)
            hit = idx.get(base, "ABSENT")
            if hit == "ABSENT" or hit is None:
                (missing if hit == "ABSENT" else ambiguous).append(name)
            else:
                r[key] = hit
    if missing or ambiguous:
        raise FileNotFoundError(
            f"AgeDB 图像解析失败: 缺失 {len(missing)}（如 {missing[:3]}）；"
            f"重名歧义 {len(ambiguous)}（如 {ambiguous[:3]}）")
    return rows


def load_tinyface_layout(root):
    """TinyFace probe-gallery 布局自适应（卡 :618 :620）：
    A) probe/ 与 gallery/ 目录树 → identity = 图像父目录名；
    B) probe.txt / gallery.txt 列表（行式 '<relpath> <identity>'）。
    返回 rows: [{'role': 'probe'|'gallery', 'p': rel, 'identity': str}, ...]。
    """
    probe_dir = gallery_dir = None
    for dirpath, dirs, _f in os.walk(root):
        for d in list(dirs):
            dl = d.lower()
            if dl == "probe" and probe_dir is None:
                probe_dir = os.path.join(dirpath, d)
            elif dl == "gallery" and gallery_dir is None:
                gallery_dir = os.path.join(dirpath, d)

    def _walk(role, base):
        out = []
        for dirpath, _d, files in os.walk(base):
            for fn in sorted(files):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    full = os.path.join(dirpath, fn)
                    identity = os.path.basename(dirpath)
                    out.append({"role": role,
                                "p": os.path.relpath(full, root),
                                "identity": identity})
        return out

    if probe_dir and gallery_dir:
        rows = _walk("probe", probe_dir) + _walk("gallery", gallery_dir)
        if rows:
            return rows

    lists = {}
    for dirpath, _d, files in os.walk(root):
        for fn in files:
            fl = fn.lower()
            if fl in ("probe.txt", "probe_list.txt"):
                lists["probe"] = os.path.join(dirpath, fn)
            elif fl in ("gallery.txt", "gallery_list.txt"):
                lists["gallery"] = os.path.join(dirpath, fn)
    if "probe" in lists and "gallery" in lists:
        rows = []
        for role in ("probe", "gallery"):
            with open(lists[role], "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(f"{lists[role]}:{ln} 行式应为 '<relpath> <identity>'，实为 {line!r}")
                    rows.append({"role": role, "p": parts[0], "identity": parts[-1]})
        if rows:
            return rows
    raise FileNotFoundError(
        f"TinyFace 布局无法识别（既无 probe/gallery 目录树，也无 probe.txt/gallery.txt）: {root}")


def validate_pairwise_rows(rows, n_folds=10):
    """结构体检：折号连续 0..n_folds-1、每折 same/diff 平衡。不平/缺折 → 硬错。"""
    folds = sorted({r["fold"] for r in rows})
    if folds != list(range(n_folds)):
        raise ValueError(f"折号应为 0..{n_folds - 1}，实为 {folds}")
    for f in folds:
        s = sum(1 for r in rows if r["fold"] == f and r["label"] == 1)
        d = sum(1 for r in rows if r["fold"] == f and r["label"] == 0)
        if s == 0 or d == 0 or s != d:
            raise ValueError(f"折 {f} 不平衡（same={s}, diff={d}），违反官方 10-fold 平衡")
    return True


def assert_official_counts(dataset, protocol, rows):
    """官方口径硬校验（禁止简化/裁对）：
    CFP-FP/CFP-FF = 7000 对（3500/3500）；AgeDB-30 = 6000 对（3000/3000）。"""
    if dataset == "cfp_fp":
        want, tag = 7000, f"CFP-{protocol}"
    elif dataset == "agedb30":
        want, tag = 6000, "AgeDB-30"
    else:
        raise ValueError(f"{dataset} 不是 pairwise 基准")
    n = len(rows)
    s = sum(1 for r in rows if r["label"] == 1)
    if n != want or s != want // 2:
        raise ValueError(f"{tag} 官方口径 = {want} 对（same={want // 2}），实际 n={n} same={s}")
    return True


def assert_kind_matches(dataset, rows):
    """反统一双保险（卡 :607-609 / :705）：数据集协议种类与行形态必须一致，否则硬错。"""
    kind = KIND_OF_DATASET[dataset]
    if not rows:
        raise ValueError("空清单")
    looks_pairwise = all(k in rows[0] for k in ("fold", "label", "p1", "p2"))
    looks_rank1 = all(k in rows[0] for k in ("role", "p", "identity"))
    if kind == "pairwise" and looks_rank1:
        raise ValueError(f"{DATASET_CANON[dataset]} 走 pairwise，行却是 probe-gallery 形态（协议错位）")
    if kind == "rank1" and looks_pairwise:
        raise ValueError("TinyFace Rank-1 协议禁止按 pairwise 评测（卡 :607-609，禁统一）")
    if kind == "pairwise" and not looks_pairwise:
        raise ValueError("pairwise 行缺 fold/label/p1/p2")
    if kind == "rank1" and not looks_rank1:
        raise ValueError("rank1 行缺 role/p/identity")
    return kind


# ============================ 评测层 ============================

def pairwise_sims(feats):
    """feats (2N,512) 按对排列 → 相似度 (N,)。公式与 lfw_eval.evaluate_lfw 逐位一致。"""
    if len(feats) % 2 != 0:
        raise ValueError("pairwise 特征数须为偶数")
    sims = []
    for i in range(len(feats) // 2):
        a, b = feats[2 * i], feats[2 * i + 1]
        sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
    return np.array(sims, dtype=np.float64)


def evaluate_pairwise_folds(sims, labels, folds, n_folds=10):
    """官方折指定的 10-fold verification（与 LFW 同一阈值搜索逻辑，测试折=官方折号）：
    每折在其余折上用 roc_curve 候选搜最优阈值（val acc 严格增者取之），本折测试；
    cv = 10 折均值。AUC/EER/TAR@FAR 全量阈值无关计算（口径同 lfw_eval）。"""
    sims = np.asarray(sims, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    folds = np.asarray(folds, dtype=int)
    if not (len(sims) == len(labels) == len(folds)):
        raise ValueError("sims/labels/folds 长度不一致")
    uniq = sorted(set(folds.tolist()))
    if uniq != list(range(n_folds)):
        raise ValueError(f"折号应为 0..{n_folds - 1}，实为 {uniq}")

    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    fold_accs, fold_thresholds = [], []
    all_idx = np.arange(len(sims))
    for f in uniq:
        test_idx = all_idx[folds == f]
        val_idx = all_idx[folds != f]
        if len(set(labels[val_idx].tolist())) < 2:
            raise ValueError(f"折 {f}: 验证集单类，无法搜阈值")
        if len(set(labels[test_idx].tolist())) < 2:
            raise ValueError(f"折 {f}: 测试集单类，官方协议要求每折 same/diff 平衡")
        val_fpr, val_tpr, val_thresh = roc_curve(labels[val_idx], sims[val_idx])
        best_acc, best_t = 0.0, 0.0
        for t in val_thresh:
            acc = float(np.mean((sims[val_idx] >= t).astype(int) == labels[val_idx]))
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        fold_acc = float(np.mean((sims[test_idx] >= best_t).astype(int) == labels[test_idx]))
        fold_accs.append(fold_acc)
        fold_thresholds.append(best_t)

    eer_idx = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    tar = {}
    for far in (1e-3, 1e-4, 1e-6):
        k = int(np.searchsorted(fpr, far))
        tar[f"tar@{far}"] = float(tpr[k]) if k < len(tpr) else 0.0

    return {
        "acc_cv": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "fold_accs": fold_accs,
        "fold_thresholds": fold_thresholds,
        "auc": float(roc_auc),
        "eer": float((fpr[eer_idx] + (1 - tpr)[eer_idx]) / 2),
        **tar,
        "n_pairs": int(len(sims)),
        "n_same": int(labels.sum()),
        "n_diff": int(len(labels) - labels.sum()),
        "n_folds": n_folds,
    }


def rank1_accuracy(S, probe_ids, gallery_ids):
    """TinyFace probe-gallery Rank-1（卡 :620，非 pairwise）：
    每个 probe 在 gallery 上取余弦 argmax，命中同身份记对。
    - 并列最大取首个（np.argmax，确定性，如实记）；
    - probe 身份不在 gallery（开放集无答案）一律记错，同时披露计数——不挑样本。
    """
    S = np.asarray(S, dtype=np.float64)
    probe_ids = list(probe_ids)
    gallery_ids = list(gallery_ids)
    if S.shape != (len(probe_ids), len(gallery_ids)):
        raise ValueError(f"相似度矩阵 {S.shape} ≠ ({len(probe_ids)},{len(gallery_ids)})")
    top = np.argmax(S, axis=1)
    hits = [gallery_ids[t] == probe_ids[i] for i, t in enumerate(top)]
    in_g = set(gallery_ids)
    n_unans = sum(1 for q in probe_ids if q not in in_g)
    return {
        "rank1": float(np.mean(hits)) if hits else 0.0,
        "n_probe": len(probe_ids),
        "n_gallery": len(gallery_ids),
        "n_probe_id_not_in_gallery": int(n_unans),
        "n_hit": int(np.sum(hits)),
    }


# ============================ 数据与嵌入 ============================

def load_manifest_rows(dataset, args):
    """按数据集构建/校验清单行（官方件→结构→官方口径硬校验）。"""
    if dataset == "cfp_fp":
        protocol = args.protocol or "FP"
        rows = load_cfp_protocol(args.protocol_root, protocol)
        validate_pairwise_rows(rows)
        assert_official_counts(dataset, protocol, rows)
        resolved = [{**r, "p1": os.path.join(args.aligned_root, r["p1"]),
                     "p2": os.path.join(args.aligned_root, r["p2"])} for r in rows]
        return protocol, resolved
    if dataset == "agedb30":
        rows = load_agedb_pairs(args.pairs_txt)
        validate_pairwise_rows(rows)
        assert_official_counts(dataset, None, rows)
        # 解析在 aligned_root 下进行（名字→真实相对路径→拼 aligned_root）
        tmp_rows = [{"p1": r["p1"], "p2": r["p2"]} for r in rows]
        resolve_agedb_names(tmp_rows, args.aligned_root)
        for r, t in zip(rows, tmp_rows):
            r["p1"] = os.path.join(args.aligned_root, t["p1"])
            r["p2"] = os.path.join(args.aligned_root, t["p2"])
        return "30", rows
    rows = load_tinyface_layout(args.aligned_root)
    for r in rows:
        r["p"] = os.path.join(args.aligned_root, r["p"])
    return "probe_gallery", rows


def write_manifest(dataset, protocol, rows, manifest_dir, sources):
    """清单落盘（确定性字节序：重生成=同 sha）。CSV + meta（源文件 sha256 溯源）。"""
    os.makedirs(manifest_dir, exist_ok=True)
    kind = KIND_OF_DATASET[dataset]
    csv_path = os.path.join(manifest_dir, f"{dataset}_manifest.csv")
    fields = ["fold", "label", "p1", "p2"] if kind == "pairwise" else ["role", "p", "identity"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    meta = {
        "dataset": DATASET_CANON[dataset],
        "protocol": protocol,
        "kind": kind,
        "n_rows": len(rows),
        "manifest_sha256": sha256_file(csv_path),
        "sources": sources,
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/eval_generic_pairs.py",
    }
    meta_path = os.path.join(manifest_dir, f"{dataset}_manifest.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return csv_path, meta


def check_images(rows, kind):
    """严格数据体检：全部引用图像必须存在且可读，缺一即退（禁止静默丢对）。"""
    if kind == "pairwise":
        paths = sorted({r["p1"] for r in rows} | {r["p2"] for r in rows})
    else:
        paths = sorted({r["p"] for r in rows})
    bad = []
    for p in paths:
        if not os.path.exists(p):
            bad.append((p, "missing"))
            continue
        img = imread_cn(p)
        if img is None or img.size == 0:
            bad.append((p, "unreadable"))
    if bad:
        print(f"[ERROR] {len(bad)}/{len(paths)} 图像缺失/不可读（如 {bad[0]}），拒绝评测", flush=True)
        sys.exit(1)
    return paths


def _preprocess(path):
    img = imread_cn(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
    return np.transpose(img, (2, 0, 1))


def extract_embeddings(paths, backend, batch_size, device):
    """backend: ('edgeface', model) | ('onnx', session)。返回 dict path→(512,) float32。"""
    cache = {}
    kind, obj = backend
    if kind == "edgeface":
        import torch
        model = obj
        with torch.no_grad():
            for s in range(0, len(paths), batch_size):
                chunk = paths[s:s + batch_size]
                batch = np.stack([_preprocess(p) for p in chunk])
                t = torch.from_numpy(batch).to(device)
                feats = model.get_embedding(t).cpu().numpy()[:, :512]
                for p, v in zip(chunk, feats):
                    cache[p] = v.astype(np.float32)
    else:
        sess = obj
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        for s in range(0, len(paths), batch_size):
            chunk = paths[s:s + batch_size]
            batch = np.stack([_preprocess(p) for p in chunk]).astype(np.float32)
            out = sess.run([oname], {iname: batch})[0]
            out = np.asarray(out).reshape(len(chunk), -1)
            for p, v in zip(chunk, out):
                cache[p] = v.astype(np.float32)
    return cache


def build_backend(args, device):
    if args.calib_onnx:
        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(args.calib_onnx,
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            sess = ort.InferenceSession(args.calib_onnx, providers=["CPUExecutionProvider"])
        print(f"[backend] onnx 校准模型（已发表权重）: {args.calib_onnx}", flush=True)
        return "onnx", sess, args.calib_onnx
    if not args.ckpt:
        print("[ERROR] 需要 --ckpt <pt> 或 --calib_onnx <onnx>", flush=True)
        sys.exit(2)
    model = build_edgeface(DEFAULT_CONFIGS[args.cfg], backbone_type=args.backbone).to(device)
    ckpt = torch_load(args.ckpt, device)
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    print(f"[backend] EdgeFace cfg={args.cfg} backbone={args.backbone} ckpt={args.ckpt}", flush=True)
    return "edgeface", model, args.ckpt


def torch_load(path, device):
    import torch
    return torch.load(path, map_location=device)


# ============================ main ============================

def main():
    ap = argparse.ArgumentParser(description="T-R5 多基准评估管线（CFP-FP / AgeDB-30 / TinyFace）")
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    ap.add_argument("--protocol", type=str, default=None, help="仅 cfp_fp：FP（默认，主协议）或 FF")
    ap.add_argument("--ckpt", type=str, default=None, help="EdgeFace checkpoint（与 --calib_onnx 二选一）")
    ap.add_argument("--cfg", type=str, default="cfg4")
    ap.add_argument("--backbone", type=str, default="shufflenet",
                    choices=["shufflenet", "pconv"])
    ap.add_argument("--calib_onnx", type=str, default=None, help="已发表模型 ONNX（校准模式）")
    ap.add_argument("--aligned_root", type=str, default=None,
                    help="112×112 对齐图根目录（默认见 DEFAULT_ALIGNED）")
    ap.add_argument("--protocol_root", type=str, default=DEFAULT_PROTOCOL_ROOT,
                    help="CFP 官方协议源目录")
    ap.add_argument("--pairs_txt", type=str, default=DEFAULT_AGEDB_PAIRS,
                    help="AgeDB-30 官方 6000 对文件")
    ap.add_argument("--manifest_dir", type=str, default=MANIFEST_DIR)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--manifest_only", action="store_true", help="只建清单+数据体检后退出")
    args = ap.parse_args()

    if args.aligned_root is None:
        args.aligned_root = DEFAULT_ALIGNED[args.dataset]
    dataset = args.dataset
    kind = KIND_OF_DATASET[dataset]

    # 拒覆盖（红线）：先挡再干活
    tag = os.path.splitext(os.path.basename(args.calib_onnx))[0] if args.calib_onnx else \
        os.path.splitext(os.path.basename(args.ckpt or "model"))[0]
    out_path = args.out or f"results/eval_{dataset}_{tag}.json"
    if os.path.exists(out_path):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}（请传 --out <新路径>）")
        sys.exit(5)
    if not args.manifest_only and not (args.ckpt or args.calib_onnx):
        print("[ERROR] 需要 --ckpt <pt> 或 --calib_onnx <onnx>（--manifest_only 除外）")
        sys.exit(2)

    # 1) 清单 + 官方口径 + 数据体检
    protocol, rows = load_manifest_rows(dataset, args)
    assert_kind_matches(dataset, rows)          # 反统一双保险
    sources = {}
    if dataset == "cfp_fp":
        pdir = _cfp_locate(args.protocol_root)
        sources["protocol_root"] = os.path.abspath(pdir)
        for fn in ("Pair_list_F.txt", "Pair_list_P.txt"):
            sources[fn] = sha256_file(os.path.join(pdir, fn))
    elif dataset == "agedb30":
        sources["pairs_txt"] = os.path.abspath(args.pairs_txt)
        sources["pairs_sha256"] = sha256_file(args.pairs_txt)
    sources["aligned_root"] = os.path.abspath(args.aligned_root)

    csv_path, meta = write_manifest(dataset, protocol, rows, args.manifest_dir, sources)
    print(f"[manifest] {csv_path}  rows={len(rows)}  sha256={meta['manifest_sha256'][:16]}…", flush=True)
    paths = check_images(rows, kind)
    print(f"[data] {len(paths)} 图像全部可读", flush=True)
    if args.manifest_only:
        print("[manifest_only] 清单+体检完成，退出")
        return

    # 2) 嵌入
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend, obj, model_path = build_backend(args, device)
    cache = extract_embeddings(paths, (backend, obj), args.batch, device)
    print(f"[embed] {len(cache)} 特征就绪（{backend}）", flush=True)

    # 3) 协议评测（按 KIND 分派，禁止互相误入）
    extra = {}
    if kind == "pairwise":
        assert_kind_matches(dataset, rows)
        feats = []
        for r in rows:
            feats.append(cache[r["p1"]])
            feats.append(cache[r["p2"]])
        feats = np.stack(feats)
        sims = pairwise_sims(feats)
        res = evaluate_pairwise_folds(sims, [r["label"] for r in rows], [r["fold"] for r in rows])
        metric, value = METRIC_OF_KIND["pairwise"], res["acc_cv"]
        extra = {"protocol_detail": protocol, "folds": res.pop("n_folds"),
                 "acc_std": res["acc_std"], "fold_accs": res["fold_accs"],
                 "fold_thresholds": res["fold_thresholds"],
                 "n_pairs": res["n_pairs"], "n_same": res["n_same"], "n_diff": res["n_diff"],
                 "auc": res["auc"], "eer": res["eer"],
                 "tar@1e-3": res["tar@1e-3"], "tar@1e-4": res["tar@1e-4"],
                 "tar@1e-6": res["tar@1e-6"]}
    else:
        assert_kind_matches(dataset, rows)
        probes = [r for r in rows if r["role"] == "probe"]
        galleys = [r for r in rows if r["role"] == "gallery"]
        if not probes or not galleys:
            print(f"[ERROR] probe={len(probes)} gallery={len(galleys)} 须均非空")
            sys.exit(1)
        fp = np.stack([cache[r["p"]] for r in probes])
        fg = np.stack([cache[r["p"]] for r in galleys])
        fp /= (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8)
        fg /= (np.linalg.norm(fg, axis=1, keepdims=True) + 1e-8)
        S = fp @ fg.T
        res = rank1_accuracy(S, [r["identity"] for r in probes], [r["identity"] for r in galleys])
        metric, value = METRIC_OF_KIND["rank1"], res["rank1"]
        extra = {"protocol_detail": protocol, **res}

    # 4) 统一 JSON（卡 :622-623 五键）
    config = {
        "dataset": DATASET_CANON[dataset],
        "protocol": protocol,
        "metric": metric,
        "input_size": 112,
        "preprocess": PREPROCESS_DESC,
        "manifest_sha256": meta["manifest_sha256"],
        "folds": 10 if kind == "pairwise" else "probe_gallery_rank1",
    }
    ch = make_config_hash(config)
    payload = build_payload(
        model_path=model_path, dataset=DATASET_CANON[dataset], metric=metric,
        value=float(value), config_hash=ch,
        protocol=protocol, kind=kind, backend=backend,
        cfg=(None if backend == "onnx" else args.cfg),
        backbone=(None if backend == "onnx" else args.backbone),
        aligned_root=os.path.abspath(args.aligned_root),
        manifest=os.path.abspath(csv_path), manifest_sha256=meta["manifest_sha256"],
        preprocess=PREPROCESS_DESC, device=device,
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        script="scripts/eval_generic_pairs.py",
        **extra,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):                 # 双保险（并发/竞态兜底）
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}")
        sys.exit(5)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[done] {DATASET_CANON[dataset]}  {metric}={value * 100:.2f}%  → {out_path}", flush=True)


if __name__ == "__main__":
    main()
