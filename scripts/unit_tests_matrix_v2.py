"""v2.0 注册矩阵单元测试（prereg_v2_matrix pin 1：T-R2/T-R3/T-R4/T-R5 + D-factor 前向）。

判据来源 = 白皮书卡原文（_wp2_extract:541-626）+ prereg §6 细则（P4 2026-10-14 复签）：
  T-R2：高范数样本 λ_i > 低范数；λ_i ∈ (0, λ_max)；梯度不通过范数计算回传（stop-grad 只读）
  T-R3：固定随机特征，手工计算距离/角度矩阵与实现输出比对（误差 < 1e-5）；
        距离归一化除零保护；三元组采样上限（4096/step）
  T-R4：双骨干参数量在卡带 0.9-1.0M；前向输出 shape (B, 512)；单批梯度反传无 NaN；
        统计件 harness（CPU perf_counter + warmup/runs 精确计数；FLOPs=thop 方法锁）
  T-R5：CFP-FP/AgeDB-30 官方 10-fold 解析 + 官方口径硬校验（7000/6000 对）；
        TinyFace probe-gallery Rank-1 未被统一成 pairwise（反例矩阵：同数据两协议值不同）；
        统一 JSON 五键 {model_path,dataset,metric,value,config_hash} + config_hash 确定性
  D 因子：d0-d4 前向全通、logs 形态齐、d4 附 g 分布（min/max/mean/p25/p75）
  G：引文保真机检（gate_check CRITERIA 三块 ↔ prereg §13/§14 原文一致）

零 GPU（CPU 即可）；退出码 0 = 全 PASS。
用法：python scripts/unit_tests_matrix_v2.py
"""
import os
import sys
import math

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from qad_face.models.edgeface import build_by_name                      # noqa: E402
from qad_face.train.losses import (ArcFaceLoss, DistillV2Loss,          # noqa: E402
                                   RKDLoss, DISTILL_TYPES, quality_cond_g)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


# ============================== T-R2 ==============================
def t_r2():
    print("[T-R2] 质量条件化 λ_i（stop-grad 只读）")
    torch.manual_seed(0)
    # ① 高范数样本 g > 低范数样本 g（同批对照）
    emb = torch.zeros(8, 512)
    emb[:4] = torch.randn(4, 512)
    emb[4:] = torch.randn(4, 512) * 0.01
    emb[0] = emb[0] / emb[0].norm() * 20.0     # 明确高范数
    emb[4] = emb[4] / (emb[4].norm() + 1e-12) * 0.5   # 明确低范数
    g = quality_cond_g(emb, alpha=1.0)
    check("T-R2-1 高范数 g > 低范数 g", bool((g[0] > g[4]).item()),
          f"g_high={float(g[0]):.4f} g_low={float(g[4]):.4f}")
    # ② λ_i ∈ (0, λ_max)
    lam_max = 0.1
    z = torch.randn(256, 512)
    lam_i = lam_max * quality_cond_g(z, alpha=1.0)
    check("T-R2-2 λ_i ∈ (0, λ_max)", bool((lam_i > 0).all() and (lam_i < lam_max).all()),
          f"min={float(lam_i.min()):.4f} max={float(lam_i.max()):.4f}")
    # ③ 梯度不通过范数回传：L = Σ λ_max·g_i·‖z_i‖² 的解析梯度 = λ_max·g_i·2z_i（g 只读）
    z = torch.randn(16, 512, requires_grad=True)
    g = quality_cond_g(z, alpha=1.0)
    L = (z.pow(2).sum(dim=1) * (lam_max * g)).sum()
    L.backward()
    expect = lam_max * g.detach().view(-1, 1) * 2.0 * z.detach()
    max_err = float((z.grad - expect).abs().max())
    check("T-R2-3 范数项 stop-grad（梯度=只读解析式）", max_err < 1e-6, f"max_err={max_err:.2e}")
    # ④ d4 前向附 g 分布日志（T-R2 #4 的 step 级来源）
    arc = ArcFaceLoss(512, 10, s=30.0, m=0.3)
    crit = DistillV2Loss(arc, "d4", lambda_max=0.1, alpha=1.0, seed=0)
    s = torch.randn(32, 512)
    t = torch.randn(32, 512)
    lab = torch.randint(0, 10, (32,))
    loss, logits, logs = crit(s, t, lab)
    keys_ok = all(k in logs for k in ("g_min", "g_max", "g_mean", "g_p25", "g_p75",
                                      "rkd_d", "rkd_a", "loss_rkd", "loss_arc", "loss_total"))
    check("T-R2-4 d4 logs 含 g 分布五统计 + rkd 分量", keys_ok and torch.isfinite(loss),
          f"loss={float(loss):.4f}")


# ============================== T-R3 ==============================
def t_r3():
    print("[T-R3] RKD-D / RKD-A（手算比对 <1e-5）")
    torch.manual_seed(1)
    rkd = RKDLoss(rkd_d_weight=1.0, rkd_a_weight=1.0, n_triplets=4096, seed=0)
    # ① 距离矩阵 vs 手算（纯 python 双重循环，B=5）
    z = torch.randn(5, 16).float()
    got = rkd.distance_matrix(z).numpy()
    za = z.numpy()
    err = 0.0
    for i in range(5):
        for j in range(5):
            manual = math.sqrt(sum((float(za[i, k] - za[j, k]) ** 2 for k in range(16))))
            err = max(err, abs(float(got[i, j]) - manual))
    check("T-R3-1 距离矩阵手算比对 <1e-5", err < 1e-5, f"max_err={err:.2e}")
    # ② 角度公式 vs 手算（arccos((e_ij·e_ik)/(|e_ij||e_ik|))，e_ij=f_i−f_j；角点=i）
    z6 = torch.randn(6, 16).float()
    i = torch.tensor([0, 3]); j = torch.tensor([1, 4]); k = torch.tensor([2, 5])
    ang = rkd.angles(z6, i, j, k).numpy()
    z6a = z6.numpy()
    err2 = 0.0
    for n, (ii, jj, kk) in enumerate([(0, 1, 2), (3, 4, 5)]):
        e_ij = z6a[ii] - z6a[jj]
        e_ik = z6a[ii] - z6a[kk]
        cos = float(np.dot(e_ij, e_ik) / (np.linalg.norm(e_ij) * np.linalg.norm(e_ik)))
        manual = math.acos(min(1.0, max(-1.0, cos)))
        err2 = max(err2, abs(float(ang[n]) - manual))
    check("T-R3-2 角度公式手算比对 <1e-5", err2 < 1e-5, f"max_err={err2:.2e}")
    # ③ 除零保护：全同 batch → 距离均值 0 → clamp 生效，loss 有限
    zdup = torch.randn(1, 16).repeat(8, 1)
    loss, logs = rkd(zdup, zdup * 0.5, w_pair=None, w_trip=None,
                     triples=(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2, 3])))
    check("T-R3-3 距离归一化除零保护（全同 batch loss 有限）", bool(torch.isfinite(loss)),
          f"loss={float(loss):.4f}")
    # ④ 三元组上限 + 无放回（i≠j、i≠k、j≠k）
    i3, j3, k3 = rkd.sample_triples(256, "cpu")
    uniq = bool((i3 != j3).all() and (i3 != k3).all() and (j3 != k3).all())
    check("T-R3-4 三元组上限 4096 + 无放回", len(i3) == 4096 and uniq, f"n={len(i3)}")
    rkd_s = RKDLoss(n_triplets=4096, seed=0)
    i3s, _, _ = rkd_s.sample_triples(5, "cpu")
    check("T-R3-5 小 batch 上限 = B(B-1)(B-2)", len(i3s) == 5 * 4 * 3, f"n={len(i3s)}")
    # ⑤ d3 等权缩放结构：分量和 = loss_rkd（权重 1.0/1.0）；λ_max 缩放在总损失侧
    torch.manual_seed(2)
    arc = ArcFaceLoss(512, 10, s=30.0, m=0.3)
    crit = DistillV2Loss(arc, "d3", lambda_max=0.1, seed=0)
    s = torch.randn(32, 512); t = torch.randn(32, 512); lab = torch.randint(0, 10, (32,))
    _, _, logs = crit(s, t, lab)
    recomputed = logs["rkd_d"] + logs["rkd_a"]
    check("T-R3-6a d3 分量和 = loss_rkd（等权 1.0/1.0）", abs(recomputed - logs["loss_rkd"]) < 1e-6,
          f"sum={recomputed:.6f} loss_rkd={logs['loss_rkd']:.6f}")
    total_minus_arc = logs["loss_total"] - logs["loss_arc"]
    check("T-R3-6b 总损失−ArcFace = λ_max·loss_rkd（全局缩放）",
          abs(total_minus_arc - 0.1 * logs["loss_rkd"]) < 1e-6,
          f"Δ={total_minus_arc:.6f} vs 0.1·rkd={0.1 * logs['loss_rkd']:.6f}")


# ============================== T-R4 ==============================
def t_r4():
    print("[T-R4] PConv 骨干（卡带 0.9-1.0M / shape / 梯度）")
    m_std = build_by_name("x05_sab_full")
    m_pc = build_by_name("x05_sab_full", backbone_type="pconv")
    p_std = sum(p.numel() for p in m_std.parameters())
    p_pc = sum(p.numel() for p in m_pc.parameters())
    check("T-R4-1 双骨干参数量 ∈ [0.9M, 1.0M]（卡带）",
          900_000 <= p_std <= 1_000_000 and 900_000 <= p_pc <= 1_000_000,
          f"std={p_std} pconv={p_pc} (Δ={p_pc - p_std:+d})")
    x = torch.randn(2, 3, 112, 112)
    m_std.eval(); m_pc.eval()
    o1 = m_std(x); o2 = m_pc(x)
    check("T-R4-2 前向 shape (B,512) 双骨干一致",
          list(o1.shape) == [2, 512] and list(o2.shape) == [2, 512],
          f"std={list(o1.shape)} pconv={list(o2.shape)}")
    for name, m in (("std", m_std), ("pconv", m_pc)):
        m.train()
        for p in m.parameters():
            p.grad = None
        loss = m(torch.randn(4, 3, 112, 112)).pow(2).mean()
        loss.backward()
        bad = sum(1 for p in m.parameters()
                  if p.grad is not None and not bool(torch.isfinite(p.grad).all()))
        check(f"T-R4-3 梯度反传无 NaN/Inf（{name}）", bad == 0, f"bad_grads={bad}")

    # --- 统计件 harness（卡 :600-602 :700；零 GPU 段——GPU Event 段由正式跑 JSON 留证）---
    import importlib.util as _ilu
    spec_s = _ilu.spec_from_file_location(
        "stats_efficiency", os.path.join(ROOT, "scripts", "stats_efficiency.py"))
    se = _ilu.module_from_spec(spec_s)
    spec_s.loader.exec_module(se)

    class _Counting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n = 0

        def forward(self, x):
            self.n += 1
            return x

    cm = _Counting()
    r44 = se.measure_cpu(cm, torch.randn(1, 3, 112, 112), runs=10, warmup=5)
    ok44 = (cm.n == 15 and r44["n"] == 10 and r44["p50"] > 0 and r44["p95"] > 0
            and set(r44) >= {"p50", "p95", "mean", "n"})
    check("T-R4-4 CPU 延迟 harness（perf_counter / warmup5+runs10 精确计数 / p50 p95）",
          ok44, f"n_calls={cm.n} stats_n={r44['n']}")

    fl45 = se.flops_thop(build_by_name("x05_sab_full"), 112)
    ok45 = (fl45["method"] == "thop" and fl45["macs"] > 0 and fl45["params"] > 0
            and abs(fl45["flops"] - 2 * fl45["macs"]) < 1e-6)
    check("T-R4-5 FLOPs=thop 口径（方法锁 / macs>0 / flops=2×macs）",
          ok45, f"macs={fl45['macs']:.3e} thop={fl45['thop_version']}")


# ============================== D 因子前向 ==============================
def t_d_forward():
    print("[D-factors] d0-d4 前向全通 + 互斥表完整性")
    torch.manual_seed(3)
    arc = ArcFaceLoss(512, 10, s=30.0, m=0.3)
    s = torch.randn(32, 512); t = torch.randn(32, 512); lab = torch.randint(0, 10, (32,))
    for dt in ("d0", "d1", "d2", "d3", "d4"):
        crit = DistillV2Loss(arc, dt, lambda_max=0.1, alpha=1.0, seed=0)
        loss, logits, logs = crit(s, t, lab)
        has_dist = any(k in logs for k in ("loss_feat", "rkd_d", "rkd_a"))
        expect_dist = dt != "d0"
        check(f"D-{dt} 前向有限 + 蒸馏分量{'有' if expect_dist else '无'}",
              bool(torch.isfinite(loss)) and (has_dist == expect_dist),
              f"loss={float(loss):.4f} keys={sorted(logs)}")
    # d0 + teacher=None（无教师路径）
    crit = DistillV2Loss(arc, "d0")
    loss, _, logs = crit(s, None, lab)
    check("D-d0 teacher=None 前向有限", bool(torch.isfinite(loss)))
    # 注册表键完备
    check("DISTILL_TYPES 键 = {d0..d4}", set(DISTILL_TYPES) == {"d0", "d1", "d2", "d3", "d4"})


def t_integration():
    print("[IT] DistillTrainer d4 + D(t) 插桩端到端（合成数据，CPU，不碰真实数据集）")
    import io
    import tempfile
    import contextlib
    from torch.utils.data import DataLoader
    from qad_face.train.distill import DistillTrainer

    class FakeDS(torch.utils.data.Dataset):
        def __init__(self, n=64):
            self.x = torch.randn(n, 3, 112, 112)
            self.labels = [i % 10 for i in range(n)]

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.labels[i], i

    ds = FakeDS(64)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    teacher = torch.randn(64, 512)
    tmp = tempfile.mkdtemp(prefix="utmv2_")
    trainer = DistillTrainer({"width_mult": 0.5, "use_sab": True}, num_classes=10,
                             device="cpu", seed=42, distill_type="d4",
                             lambda_max=0.1, alpha=1.0, n_triplets=64,
                             backbone_type="pconv")
    dt_idxs = torch.arange(0, 64, 4)
    dt_labels = torch.tensor([ds.labels[int(i)] for i in dt_idxs])
    buf = io.StringIO()
    ok_train = True
    try:
        with contextlib.redirect_stdout(buf):
            trainer.train(dl, teacher, epochs=1, lr=0.05, save_dir=tmp,
                          save_name="smoke_d4", log_every=4, use_amp=False,
                          grad_clip=10.0, warmup_epochs=1, distill_start_epoch=0,
                          dt_idxs=dt_idxs, dt_labels=dt_labels, dt_every=4)
    except Exception as e:
        ok_train = False
        err = repr(e)
    out = buf.getvalue()
    dt_line = any(l.startswith("  [Dt step 4]") for l in out.splitlines())
    lam_line = any("[lambda_i ep1]" in l for l in out.splitlines())
    arm_line = any("[step 4]" in l and "g_mean" in l for l in out.splitlines())
    ckpt_ok = os.path.exists(os.path.join(tmp, "smoke_d4_final.pt"))
    check("IT-1 d4 训练 1 epoch 全链路（PConv+RKD+λ_i）无异常", ok_train,
          "" if ok_train else err)
    check("IT-2 [Dt step 8] 真实前向行入日志", dt_line)
    check("IT-3 [lambda_i ep1] epoch 分布行入日志", lam_line)
    check("IT-4 [step] 行含 g_mean（T-R2 #4 step 级）", arm_line)
    check("IT-5 final ckpt 落盘（健康门通过）", ckpt_ok)


def t_r5():
    """T-R5 多基准评估管线（卡 :606-626 + :705；pin 1 的 TinyFace 协议件）。"""
    import tempfile
    import shutil
    import importlib.util
    import cv2

    spec = importlib.util.spec_from_file_location(
        "eval_generic_pairs", os.path.join(ROOT, "scripts", "eval_generic_pairs.py"))
    eg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eg)

    tmp = tempfile.mkdtemp(prefix="t5_")
    try:
        # ---------- CFP 官方协议夹具 ----------
        proto = os.path.join(tmp, "Protocol")
        for pd in ("FP", "FF"):
            for fd in ("01", "02"):
                os.makedirs(os.path.join(proto, "Split", pd, fd))
        with open(os.path.join(proto, "Pair_list_F.txt"), "w") as f:
            f.write("1 ../Data/Images/001/frontal/01.jpg\n2 ../Data/Images/001/frontal/02.jpg\n"
                    "3 ../Data/Images/002/frontal/01.jpg\n4 ../Data/Images/002/frontal/02.jpg\n")
        with open(os.path.join(proto, "Pair_list_P.txt"), "w") as f:
            f.write("1 ../Data/Images/001/profile/01.jpg\n2 ../Data/Images/002/profile/01.jpg\n")
        splits = {
            ("FP", "01", "same"): "1,1", ("FP", "01", "diff"): "1,2",
            ("FP", "02", "same"): "2,1", ("FP", "02", "diff"): "2,2",
            ("FF", "01", "same"): "1,2", ("FF", "01", "diff"): "1,3",
            ("FF", "02", "same"): "3,4", ("FF", "02", "diff"): "3,2",
        }
        for (pd, fd, fn), body in splits.items():
            with open(os.path.join(proto, "Split", pd, fd, f"{fn}.txt"), "w") as f:
                f.write(body + "\n")

        rows_fp = eg.load_cfp_protocol(tmp, "FP")
        ok1 = (len(rows_fp) == 4 and {r["fold"] for r in rows_fp} == {0, 1}
               and sum(r["label"] for r in rows_fp) == 2
               and rows_fp[0]["p1"].endswith("001/frontal/01.jpg")
               and rows_fp[0]["p2"].endswith("001/profile/01.jpg"))   # FP 行：F[a], P[b]
        check("T5-1 CFP-FP 官方解析（折/标签/正面-侧面路径）", ok1,
              f"n={len(rows_fp)} p1={rows_fp[0]['p1']} p2={rows_fp[0]['p2']}")

        rows_ff = eg.load_cfp_protocol(tmp, "FF")
        ok2 = all(r["p2"].split("/")[-2] == "frontal" for r in rows_ff)   # FF 双侧走 F 表
        check("T5-2 CFP-FF 双侧均为正面（F 表解析）", ok2,
              f"sample={rows_ff[0]['p2']}")

        ok3a = eg.validate_pairwise_rows(rows_fp, n_folds=2)
        raised = False
        bad2 = [{"fold": 0, "label": 1, "p1": "a", "p2": "b"},
                {"fold": 0, "label": 1, "p1": "a", "p2": "b"},
                {"fold": 1, "label": 1, "p1": "a", "p2": "b"},
                {"fold": 1, "label": 0, "p1": "a", "p2": "b"}]   # 折 0 双 same → 不平衡
        try:
            eg.validate_pairwise_rows(bad2, n_folds=2)
        except ValueError:
            raised = True
        check("T5-3 折结构平衡校验（平衡放行/不平衡拦截）", bool(ok3a and raised))

        r4 = False
        try:
            eg.assert_official_counts("cfp_fp", "FP", rows_fp)   # 4 ≠ 7000 必抛
        except ValueError:
            r4 = True
        syn7000 = [{"fold": f, "label": 1 if j < 350 else 0, "p1": "x", "p2": "y"}
                   for f in range(10) for j in range(700)]
        ok4b = eg.assert_official_counts("cfp_fp", "FP", syn7000)   # 合成官方口径放行
        check("T5-4 官方口径硬校验（夹具拦截 + 7000/3500 放行）", bool(r4 and ok4b))

        # ---------- AgeDB-30 ----------
        ap = os.path.join(tmp, "agedb_30.txt")
        with open(ap, "w") as f:
            for i in range(6000):
                label = 1 if (i % 600) < 300 else 0
                f.write(f"a{i}.jpg b{i}.jpg {label}\n")
        rows_ag = eg.load_agedb_pairs(ap)
        ok5 = (len(rows_ag) == 6000 and {r["fold"] for r in rows_ag} == set(range(10))
               and rows_ag[0]["fold"] == 0 and rows_ag[599]["fold"] == 0
               and rows_ag[600]["fold"] == 1
               and sum(r["label"] for r in rows_ag[:600]) == 300
               and eg.validate_pairwise_rows(rows_ag)
               and eg.assert_official_counts("agedb30", None, rows_ag))
        check("T5-5 AgeDB 折公式（i//600）+ 官方 6000/3000 口径 + 平衡", ok5,
              f"n={len(rows_ag)} folds={min(r['fold'] for r in rows_ag)}..{max(r['fold'] for r in rows_ag)}")

        ap2 = os.path.join(tmp, "agedb_neg.txt")
        with open(ap2, "w") as f:
            for i in range(10):
                f.write(f"a.jpg b.jpg {'-1' if i % 2 else '1'}\n")
        rows_neg = eg.load_agedb_pairs(ap2)
        ok6a = all(r["label"] in (0, 1) for r in rows_neg) and rows_neg[1]["label"] == 0
        ok6b = False
        with open(os.path.join(tmp, "agedb_bad.txt"), "w") as f:
            f.write("\n".join("a.jpg b.jpg 2" for _ in range(10)))
        try:
            eg.load_agedb_pairs(os.path.join(tmp, "agedb_bad.txt"))
        except ValueError:
            ok6b = True
        check("T5-6 AgeDB label 严解析（-1→0 放行 / 非法值拦截）", bool(ok6a and ok6b))

        img_root = os.path.join(tmp, "agedb_imgs")
        os.makedirs(os.path.join(img_root, "shardX"))
        cv2.imwrite(os.path.join(img_root, "shardX", "z.jpg"),
                    np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(img_root, "w.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
        res_rows = [{"p1": "old/z.jpg", "p2": "w.jpg"}]      # p1 走 basename 索引，p2 直拼
        eg.resolve_agedb_names(res_rows, img_root)
        ok7a = res_rows[0]["p1"] == os.path.join("shardX", "z.jpg") and res_rows[0]["p2"] == "w.jpg"
        ok7b = False
        try:
            eg.resolve_agedb_names([{"p1": "nope.jpg", "p2": "w.jpg"}], img_root)
        except FileNotFoundError:
            ok7b = True
        check("T5-7 AgeDB 图像解析（直拼+basename 索引 / 缺失硬错）", bool(ok7a and ok7b),
              f"p1={res_rows[0]['p1']}")

        # ---------- TinyFace 布局 ----------
        tf = os.path.join(tmp, "tinyA")
        for rel in ("probe/idA/p1.jpg", "probe/idB/p1.jpg", "gallery/idA/g1.jpg",
                    "gallery/idA/g2.jpg", "gallery/idB/g1.jpg", "gallery/idC/g1.jpg"):
            os.makedirs(os.path.dirname(os.path.join(tf, rel)), exist_ok=True)
            cv2.imwrite(os.path.join(tf, rel), np.zeros((8, 8, 3), dtype=np.uint8))
        rows_a = eg.load_tinyface_layout(tf)
        ok8 = (sum(r["role"] == "probe" for r in rows_a) == 2
               and sum(r["role"] == "gallery" for r in rows_a) == 4
               and {r["identity"] for r in rows_a if r["role"] == "probe"} == {"idA", "idB"}
               and eg.assert_kind_matches("tinyface", rows_a) == "rank1")
        check("T5-8 TinyFace 布局A（probe/gallery 目录树，identity=父目录）", ok8,
              f"probe={sum(r['role']=='probe' for r in rows_a)} gallery={sum(r['role']=='gallery' for r in rows_a)}")

        tfb = os.path.join(tmp, "tinyB")
        os.makedirs(tfb)
        with open(os.path.join(tfb, "probe.txt"), "w") as f:
            f.write("p/id1_0.jpg id1\np/id2_0.jpg id2\n")
        with open(os.path.join(tfb, "gallery.txt"), "w") as f:
            f.write("g/id1_1.jpg id1\ng/id2_1.jpg id2\ng/id3_1.jpg id3\n")
        rows_b = eg.load_tinyface_layout(tfb)
        ok9 = (len(rows_b) == 5 and rows_b[0] == {"role": "probe", "p": "p/id1_0.jpg",
                                                  "identity": "id1"})
        check("T5-9 TinyFace 布局B（probe/gallery 列表文件）", ok9, f"n={len(rows_b)}")

        # ---------- 反统一核心（卡 :607-609 :705）----------
        # 矩阵设计：两 probe 的 argmax 均落在异身份 gallery → Rank-1=0；
        # 同一数字拉平成 pair 视角做阈值判定（候选=观测相似度，与 roc_curve 去重集一致）→ 参照=0.5。
        S = np.array([[0.9, 0.95], [0.85, 0.5]])
        r1 = eg.rank1_accuracy(S, ["A", "B"], ["A", "B"])
        ref_sims = np.array([0.9, 0.95, 0.85, 0.5])      # (pA,gA) (pA,gB) (pB,gA) (pB,gB)
        ref_lab = np.array([1, 0, 0, 1])
        ref_acc = max(float(np.mean((ref_sims >= t).astype(int) == ref_lab))
                      for t in ref_sims)                  # 手算参照：观测候选最优阈值 acc
        ok10 = (abs(r1["rank1"] - 0.0) < 1e-12 and abs(ref_acc - 0.5) < 1e-12
                and r1["rank1"] != ref_acc)
        check("T5-10 反统一反例（同数据 Rank-1=0.0 ≠ pairwise 参照=0.5）", ok10,
              f"rank1={r1['rank1']} pair_ref={ref_acc}")

        ok11 = True
        try:
            eg.assert_kind_matches("tinyface", rows_fp)      # rank1 数据集 ← pairwise 形态
            ok11 = False
        except ValueError:
            pass
        try:
            eg.assert_kind_matches("cfp_fp", rows_a)         # pairwise 数据集 ← rank1 形态
            ok11 = False
        except ValueError:
            pass
        check("T5-11 协议错位双向拦截（assert_kind_matches）", ok11)

        ok12 = (eg.KIND_OF_DATASET == {"cfp_fp": "pairwise", "agedb30": "pairwise",
                                       "tinyface": "rank1"}
                and eg.METRIC_OF_KIND == {"pairwise": "acc_cv10", "rank1": "rank1"}
                and eg.DATASET_CANON["tinyface"] == "TinyFace")
        check("T5-12 协议-指标常量锁（tinyface=rank1/acc_cv10 不可漂移）", ok12)

        # ---------- 官方折机制 ----------
        # 夹具用类内恒定值（same=0.90 / diff=0.10）：跨折波动会触碰 LFW 式阈值搜索的
        # 边界效应（best_t 落在观测值上，边界折可失 1 样本）——那是协议固有行为，非本件机制。
        sims, labels, folds = [], [], []
        for f in range(10):
            for j in range(4):
                folds.append(f)
                labels.append(1 if j < 2 else 0)
                sims.append(0.90 if j < 2 else 0.10)
        r13 = eg.evaluate_pairwise_folds(np.array(sims), np.array(labels), np.array(folds))
        ok13a = (r13["n_folds"] == 10 and len(r13["fold_accs"]) == 10
                 and abs(r13["acc_cv"] - 1.0) < 1e-12 and abs(r13["auc"] - 1.0) < 1e-12)
        ok13b = False
        try:
            bad_l = np.array(labels)
            bad_l[np.array(folds) == 0] = 1               # 折 0 单类
            eg.evaluate_pairwise_folds(np.array(sims), bad_l, np.array(folds))
        except ValueError:
            ok13b = True
        check("T5-13 官方折 10-fold 机制（可分 cv=1.0 / 单类折拦截）", bool(ok13a and ok13b),
              f"cv={r13['acc_cv']:.3f} folds={r13['n_folds']}")

        # ---------- 严格数据体检 ----------
        rows_abs = [{**r, "p1": os.path.join(tmp, r["p1"]),
                     "p2": os.path.join(tmp, r["p2"])} for r in rows_fp]
        for r in rows_abs:
            for p in (r["p1"], r["p2"]):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                if not os.path.exists(p):
                    cv2.imwrite(p, np.zeros((8, 8, 3), dtype=np.uint8))
        ok14a = len(eg.check_images(rows_abs, "pairwise")) == 4
        victim = rows_abs[0]["p1"]
        os.remove(victim)
        ok14b = False
        try:
            eg.check_images(rows_abs, "pairwise")
        except SystemExit as e:
            ok14b = (e.code == 1)
        finally:
            cv2.imwrite(victim, np.zeros((8, 8, 3), dtype=np.uint8))
        check("T5-14 图像严格体检（可读计数 / 缺图 SystemExit(1)）", bool(ok14a and ok14b))

        # ---------- 统一 JSON（卡 :622-623）----------
        pl = eg.build_payload(model_path="m.pt", dataset="TinyFace", metric="rank1",
                              value=0.5, config_hash="h0")
        keys5 = list(pl.keys())[:5]
        ok15 = keys5 == ["model_path", "dataset", "metric", "value", "config_hash"]
        check("T5-15 统一 JSON 五键齐且键序固定", ok15, f"keys={keys5}")

        cfg_a = {"dataset": "CFP-FP", "protocol": "FP", "metric": "acc_cv10",
                 "manifest_sha256": "aaa"}
        cfg_b = dict(cfg_a)
        cfg_c = dict(cfg_a, manifest_sha256="bbb")
        ok16 = (eg.make_config_hash(cfg_a) == eg.make_config_hash(cfg_b)
                and eg.make_config_hash(cfg_a) != eg.make_config_hash(cfg_c)
                and len(eg.make_config_hash(cfg_a)) == 64)
        check("T5-16 config_hash 确定性 + 清单敏感性（sha256）", ok16)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_gate_fidelity():
    """引文保真机检（pin 5 配套）：gate_check CRITERIA 三块 ↔ prereg §13/§14 原文一致。"""
    import importlib.util as _ilu
    scripts_dir = os.path.join(ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = _ilu.spec_from_file_location("gate_check", os.path.join(scripts_dir, "gate_check.py"))
    gc = _ilu.module_from_spec(spec)
    spec.loader.exec_module(gc)
    with open(os.path.join(ROOT, "results", "prereg_v2_matrix.md"), encoding="utf-8") as f:
        prereg = f.read()

    check("G-1 CRITERIA_13 == prereg §13 原文逐字子串",
          gc.CRITERIA_13 in prereg, f"len={len(gc.CRITERIA_13)}")

    check("G-2 CRITERIA_DG_R1 == prereg §14 DG-R1 全文逐字子串",
          gc.CRITERIA_DG_R1 in prereg, f"len={len(gc.CRITERIA_DG_R1)}")

    def _norm(s):   # 表格行：去 **、竖线转空格、折叠空白（不改措辞）
        return " ".join(s.replace("**", "").replace("|", " ").split())

    row = next((ln for ln in prereg.splitlines() if ln.startswith("| **DG-R0a**")), "")
    ok3 = bool(row) and _norm(gc.CRITERIA_DG_R0A) in _norm(row)
    check("G-3 CRITERIA_DG_R0A == §14 行（去标记/折叠空白后逐字子串）", ok3,
          f"row_found={bool(row)}")


def main():
    print("=== unit_tests_matrix_v2（零 GPU）===", flush=True)
    t_r2()
    t_r3()
    t_r4()
    t_d_forward()
    t_integration()
    t_r5()
    t_gate_fidelity()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"=== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS，{n_fail} FAIL ===", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
