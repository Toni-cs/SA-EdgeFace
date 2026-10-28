# -*- coding: utf-8 -*-
"""LFW / AgeDB-30 镜像忠实性校准腿评测（prereg_mirror §7 修订 A1；诊断腿非判键腿）。

定位：镜像栈忠实性阳性对照——LFW/AgeDB 两 PASS 腿的登记事实不动、不重判（D-44/D-45
照旧），本腿只回答一个问题：「MTCNN 镜像对齐栈 + 官方协议移植」是否保得住已知好基准。
判读（§7-A1 事前写死）：
  lfw   ≥ 99.20（锚 99.70−0.5）→ 栈忠实；< 99.20 → 栈忠实性 FAIL → 回滚排查移植代码
        （优先级高于一切 TinyFace 解读）。
  agedb30 ≥ 96.08（锚 96.58−0.5）→ 同判。
栈 = 与 CFP 腿同一实现：w600k_mbf.onnx（锚 buffalo_s 行件、LFW/AgeDB 臂 1 同件同后端）
  flip-TTA 求和（归一化前，verification_evaluator.py:102）→ sklearn.preprocessing.normalize
  （L2，:103）→ calculate_roc（verification.py:27-87 逐字移植，自 _eval_cfp_mirror_v1
  同源 import，零复制漂移）LFold(10, shuffle=False) 阈值网格 arange(0,4,0.01) 折内择优。
协议件：lfw = datasets/pairs.txt（sha256 ea42330c…，6,000 对，fold-major 300same+300diff
  ×10 已核，KFold 连续块 = 官方折）；agedb30 = datasets/raw/agedb_30/agedb_30_index_pairs.txt
  （6,000 对，fold 聚集已核，resolve 硬错于缺图/歧义）。
注意（照实披露）：臂 1 的 LFW 99.2167 / AgeDB 96.5667 评测栈 = 无 flip-TTA、无 L2 行归一
  （insightface_lfw.py / egp 直乘）；本校准腿按统一镜像栈含两者 → 校准值与臂 1 值的差异
  同时含「对齐栈 + 协议栈」两变量，判据只对锚带、不对臂 1（§7-A1 字面）。
输出拒覆盖（--out 已存在即 exit 5）。
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import sklearn.preprocessing  # noqa: E402
import eval_generic_pairs as egp  # noqa: E402
from _eval_cfp_mirror_v1 import (ONNX, calculate_roc, sha256_of,  # noqa: E402  # 同源 import：零复制漂移
                                 pair_split_normalize, canary_pair_split)

PAIRS_LFW = "datasets/pairs.txt"
PAIRS_AGEDB = "datasets/raw/agedb_30/agedb_30_index_pairs.txt"
GATE = {"lfw": (99.20, 99.70), "agedb30": (96.08, 96.58)}


def load_protocol(dataset, root):
    """返回 (rels, rows=[(rel1, rel2, label_bool)])。折聚集硬断言。"""
    if dataset == "lfw":
        from qad_face.evaluation.lfw_eval import load_pairs, get_image_path
        pairs = load_pairs(PAIRS_LFW)
        assert len(pairs) == 6000, "LFW 对数非 6,000: %d" % len(pairs)
        lab = np.array([1 if p[4] else 0 for p in pairs])
        blocks = lab.reshape(10, 600)
        assert all((b[:300] == 1).all() and (b[300:] == 0).all() for b in blocks), \
            "LFW pairs 非 fold-major 300same+300diff×10（LFold 连续块前提破坏）"
        rows = []
        needed = set()
        for n1, i1, n2, i2, s in pairs:
            r1 = os.path.relpath(get_image_path(root, n1, i1), root)
            r2 = os.path.relpath(get_image_path(root, n2, i2), root)
            needed.add(r1)
            needed.add(r2)
            rows.append((r1, r2, bool(s)))
        rels = sorted(needed)
        assert len(rels) == 7701, "LFW 引用唯一图非 7,701: %d" % len(rels)
        return rels, rows
    elif dataset == "agedb30":
        rows0 = egp.load_agedb_pairs(PAIRS_AGEDB)
        assert len(rows0) == 6000, "AgeDB 对数非 6,000: %d" % len(rows0)
        folds = [r["fold"] for r in rows0]
        assert folds == sorted(folds) and [folds.count(i) for i in range(10)] == [600] * 10, \
            "AgeDB 对文件非折聚集 600×10（LFold 连续块前提破坏）"
        egp.resolve_agedb_names(rows0, root)  # 硬错于缺图/歧义
        rows = [(r["p1"], r["p2"], bool(r["label"])) for r in rows0]
        rels = sorted({r[0] for r in rows} | {r[1] for r in rows})
        assert len(rels) == 12000, "AgeDB 引用唯一图非 12,000: %d" % len(rels)
        return rels, rows
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser(description="LFW/AgeDB 镜像忠实性校准腿（prereg §7-A1）")
    ap.add_argument("--dataset", required=True, choices=["lfw", "agedb30"])
    ap.add_argument("--root", default=None, help="缺省 datasets/<ds>_mirror_112 对应根")
    ap.add_argument("--out", default=None, help="缺省 results/<ds>_mirror_calib_20261017_v1.json")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="学生臂 .pt（EdgeFace）；给定时走 torch 栈 + 四五六钉自查，否则走 onnx 教师")
    ap.add_argument("--cfg", type=str, default="x05_baseline")
    ap.add_argument("--backbone", type=str, default="shufflenet")
    args = ap.parse_args()

    root = args.root or ("datasets/lfw_mirror_112" if args.dataset == "lfw"
                         else "datasets/agedb_mirror_112")
    out = args.out or "results/%s_mirror_calib_20261017_v1.json" % args.dataset
    if os.path.exists(out):
        print("[ERROR] 目标文件已存在，拒绝覆盖: %s" % out, flush=True)
        sys.exit(5)
    # 金丝雀两测（A2 条款：两测全过才解锁计分；任一不过 → 拒绝出数、仪器回修）
    ok, canary_msg = canary_pair_split()
    print("[canary] %s" % canary_msg, flush=True)
    if not ok:
        print("[ERROR] 金丝雀不过，拒绝出数（prereg_mirror 修订 A2）", flush=True)
        sys.exit(7)
    import torch  # noqa: E402
    use_ckpt = args.ckpt is not None
    if use_ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        _fe = ck.get("epoch")
        if _fe is None or int(_fe) != 30:
            print("[ERROR] 第四钉: final_epoch=%s != 30（未完成训练）" % _fe, flush=True); sys.exit(8)
        _wv = []
        for _k, _v in ck.get("student", {}).items():
            if torch.is_floating_point(_v):
                _t = _v.detach().double().flatten(); _f = _t[torch.isfinite(_t)]
                if _f.numel(): _wv.append(_f.abs())
        _a = torch.cat(_wv); _med = float(_a.median()); _frac = float((_a < 1e-30).double().mean())
        if not (_med >= 1e-8 and _frac <= 0.01):
            print("[ERROR] 第五钉: med=%.3e frac=%.4f（湮灭臂）" % (_med, _frac), flush=True); sys.exit(9)
        from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
        model = build_edgeface(DEFAULT_CONFIGS[args.cfg], backbone_type=args.backbone).to("cpu")
        model.load_state_dict(ck["student"], strict=False); model.eval()
        device_str = "torch EdgeFace cfg=%s ckpt=%s (med=%.3e)" % (args.cfg, args.ckpt, _med)
        sess = None; iname = oname = None
    else:
        assert os.path.exists(ONNX), "w600k_mbf.onnx 缺失: %s" % ONNX
        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(
                ONNX, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        device_str = "onnxruntime %s providers=%s" % (ort.__version__, sess.get_providers())
    print("[device] %s" % device_str, flush=True)

    rels, rows = load_protocol(args.dataset, root)
    missing = [r for r in rels if not os.path.exists(os.path.join(root, r))]
    assert not missing, "镜像树缺图（首5）: %s" % missing[:5]
    n_same = sum(1 for _r1, _r2, s in rows if s)
    print("[pairs] %s = 6,000（same=%d diff=%d） 引用唯一图=%d"
          % (args.dataset, n_same, 6000 - n_same, len(rels)), flush=True)

    cache = {}
    t0 = time.time()
    for s in range(0, len(rels), args.batch):
        chunk = rels[s:s + args.batch]
        batch = np.stack([egp._preprocess(os.path.join(root, p)) for p in chunk])
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if use_ckpt:
            with torch.no_grad():
                f = model.get_embedding(torch.from_numpy(batch))[:, :512].numpy()
                fz = model.get_embedding(torch.from_numpy(
                    np.ascontiguousarray(batch[:, :, :, ::-1])))[:, :512].numpy()
            outv = np.asarray(f, dtype=np.float32).reshape(len(chunk), -1) + \
                np.asarray(fz, dtype=np.float32).reshape(len(chunk), -1)
        else:
            f = sess.run([oname], {iname: batch})[0]
            fz = sess.run([oname], {iname: np.ascontiguousarray(batch[:, :, :, ::-1])})[0]
            outv = np.asarray(f, dtype=np.float32).reshape(len(chunk), -1) + \
                np.asarray(fz, dtype=np.float32).reshape(len(chunk), -1)
        for p, v in zip(chunk, outv):
            cache[p] = v
        done = s + len(chunk)
        print("[embed] %d/%d  %.1f img/s" %
              (done, len(rels), done / max(time.time() - t0, 1e-9)), flush=True)

    e1 = np.stack([cache[r1] for r1, _r2, _s in rows])
    e2 = np.stack([cache[r2] for _r1, r2, _s in rows])
    e1n, e2n = pair_split_normalize(e1, e2)  # 官方交错布局唯一实现（D-55 修复收口）
    issame = np.asarray([s for _r1, _r2, s in rows], dtype=bool)
    thresholds = np.arange(0, 4, 0.01)
    accuracy = calculate_roc(thresholds, e1n, e2n, issame, nrof_folds=10) * 100
    acc, std = float(np.mean(accuracy)), float(np.std(accuracy))
    gate_lo, anchor = GATE[args.dataset]
    verdict = "FAITHFUL" if acc >= gate_lo else "FIDELITY_FAIL"
    print("[fold_acc] " + " ".join("%.4f" % a for a in accuracy), flush=True)
    print("[result] acc=%.4f%%  std=%.4f  gate>=%.2f  verdict=%s"
          % (acc, std, gate_lo, verdict), flush=True)

    payload = {
        "leg": "镜像忠实性校准腿（诊断腿，非判键腿；两 PASS 腿登记事实不动）",
        "dataset": args.dataset,
        "canary": canary_msg,
        "model_path": (args.ckpt if use_ckpt else ONNX),
        "model_sha256": (sha256_of(args.ckpt) if use_ckpt else sha256_of(ONNX)),
        "tta": "flip-TTA 求和（与锚同管线）",
        "crop_chain": "lfw_mirror_v1",
        "backend": device_str,
        "preprocess": egp.PREPROCESS_DESC,
        "metric": "verification acc 10-fold（官方 LFold 移植 + flip-TTA 求和 + L2）",
        "value": acc,
        "std": std,
        "fold_acc_pct": [float(a) for a in accuracy],
        "anchor": anchor,
        "gate_lo": gate_lo,
        "verdict": verdict,
        "align_root": os.path.abspath(root),
        "align_script": "scripts/_align_mirror_calib_v1.py（process_one 同源自 _align_mirror_v1 冻结件）",
        "eval_script": "scripts/_eval_calib_mirror_v1.py（calculate_roc 同源自 _eval_cfp_mirror_v1）",
        "protocol": "官方镜像栈：flip-TTA 求和 + sklearn L2 normalize + LFold(10,shuffle=False) "
                    "阈值折内择优（与 CFP 腿同实现）",
        "prereg": "results/prereg_mirror.md §7 修订 A1 + A2（先冻结后开跑）",
        "arm1_note": "臂 1 栈无 flip-TTA/无 L2 行归一（insightface_lfw.py / egp 直乘）；"
                     "本腿为统一镜像栈，与臂 1 值差异含对齐+协议两变量；判据只对锚带不对臂 1",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if os.path.exists(out):
        print("[ERROR] 目标文件已存在，拒绝覆盖: %s" % out, flush=True)
        sys.exit(5)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("[done] -> %s" % out, flush=True)


if __name__ == "__main__":
    main()
