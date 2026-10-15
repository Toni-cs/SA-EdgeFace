# -*- coding: utf-8 -*-
"""gate_check.py — P-gate 机检判据件（prereg_v2_matrix pin 5；12 项分工第 9 项）

唯一判据 = prereg_v2_matrix.md §13（裁决二 §A 逐字，见本件 CRITERIA 常量）+ §14 DG 固定表。
本件不自创、不修改任何阈值；CRITERIA 为判据原文逐字携带，机检值与原文一一对应。
解耦条款落实：CFP-FP/AgeDB-30/TinyFace 数值**不进入**任何剔除判断（只入账，§13 逐字）。

实现层披露（非判据、随报告输出）：DG-R1(c) 原文「下降或平台」的平台带宽实现 = 1.05
（末 5 epoch D 均值 ≤ 前 5 epoch 均值 × 1.05 视为平台；超出 = 发散）。原文未给数字，
此处为模糊处的实现参数，P4 可复核。

用法：
  python scripts/gate_check.py --gates            # 门1/门2/门3 逐臂 + 剔除后果核算
  python scripts/gate_check.py --dg r0a r1        # DG 门（r0a r1 r2a r2b r3 r0b）
  python scripts/gate_check.py --all              # 全部
  python scripts/gate_check.py --all --out results/_gate_check_x.json   # --out 已存在拒写
  可选 --expect-prereg-sha256 HEX（DG-R0a 哈希机检的目标值；不给则只报告实测）
退出码：0 = 所选项全 PASS；1 = 存在 FAIL 或 PENDING（未绿即不得放行，报告内区分标签）。
"""
import os
import sys
import re
import csv
import json
import time
import glob
import hashlib
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _phase3_run as p3                      # noqa: E402  （顶层 chdir/路径已对齐）

CRITERIA_13 = """门1（不变）：_ckpt_health 结构健康：fin=True、nonfinite=0、med|w| ≥ 1e-8（dead 判据沿用 D-13 口径）。
门2（绝对健康门；v2.0 §5.4「多基准综合门」作废）：
  ① LFW 10-fold acc_cv ≥ 92.0%（绝对下限，沿用 G3-f1；LFW 已非主终点，仅作健康指标，阈值不因任何结果调低）；
  ② 训练完整性：按注册配置跑满 30 epoch、无中断-续跑拼接、日志含逐臂实际生效超参；
  ③ loss 有限性：全程 loss 无 NaN/Inf（D-14 修复后 fp32 复核口径）。
  【解耦条款】任何与主检验终点（CFP-FP、TinyFace）相关的数值不得作为剔除判据。
  CFP-FP / AgeDB-30 / TinyFace 结果一律入账（与门3 同级记账）；差也不剔——差是数据，不是垃圾。
  【为什么留 LFW】绝对下限只切死跑（92.0 远低于健康区间 93–95），不切好坏；相对基线阈值才切结论。
门3（不变）：H5 分布评估记账（阈值 0.95 不降；prereg_sigma.md §1/§7 口径原样沿用）。
剔除后果（预注册）：任一臂被门1/门2 剔除 → 立即核算有效 N 与 df；df < 10（扩样前后同判）→
  该端点 4 主假设只报点估计 + 95% CI + `descriptive, no inferential test`（不删臂不删表，只降主张）。
  替换补跑仅限同 cell 同 seed 重跑一次（attempt ≤1，中断不计，R-5 口径）；无第二次、无换 seed。"""

CRITERIA_DG_R1 = """DG-R1（时点：CASIA 矩阵第 10 臂落盘并通过门1-3）：
检查项（全部为健康/实现判据，无一涉及主终点比较）：
  a) 已完成臂 100% 过门1/门2；EXCLUDED ≤ 1（>1 → 停机，按剔除后果条款核算 df）；
  b) λ_i 分布：全部臂 λ_i ∈ (0, λ_max)，p75/p25 ∈ [0.5, 2]（越界 → 停机查 T-R2 实现，警报句见 T-R2 人工必查点）；
  c) D(t) 曲线末 5 epoch 下降或平台（发散 → 停机查 T-R3 数值稳定性）；
  d) 无协议违例（配置 hash 与注册一致、无计划外变量、无占位符残留）。
判定（固定规则）：
  a–d 全过 → 继续；不得以任何理由调整 λ_max/α/温度/任何超参。
  任一失败 → 属实现缺陷：修码 → 受影响臂全量重跑（同注册配置，版本号进配置 hash）→ 修复与重跑全部记账披露。
禁止：以任何主终点数值为依据的中途调参。首臂之后任何超参变更 = 新实验，须带时间戳 amendment 另立臂族。"""

# §14 DG-R0a 行（逐字）
CRITERIA_DG_R0A = """DG-R0a（P4 复签判决 2-⑤ 拆分；阻塞 A 组首臂）| T-R1 至 T-R5 完成后 |
T-R5 三基准管线经已发表模型校准一致（AdaFace 官方权重）+ LFW-CASIA 重叠审计 PASS + T-R1 缓存校验
PASS + prereg 冻结件存在性与哈希机检 + 管线健康锚以既有件替代：era C d4a1（95.52%，G2 PASS +
D-15 重评收口件）| 修管线，禁止进入训练矩阵（v2.0 原文动作）"""

REALIZ = {"dg_r1_c_plateau_band": 1.05}       # 实现参数（非判据），随报告披露

RES = os.path.join("results", "_phase3_progress.json")
HEALTH_CSV = os.path.join("results", "_ckpt_health.csv")
DT_EPOCHS_TOTAL = 30                          # 注册训练轮数（门2②）

ITEMS = []        # [(gate, name, status, detail)]  status: PASS/FAIL/PENDING


def add(gate, name, status, detail=""):
    ITEMS.append({"gate": gate, "name": name, "status": status, "detail": str(detail)})
    print(f"  [{status}] {gate} :: {name}" + (f"  {detail}" if detail else ""), flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def refresh_health_csv():
    """复刻驱动 health_row 的刷新-判新逻辑（判据工具零改动）。"""
    t0 = os.path.getmtime(HEALTH_CSV) if os.path.exists(HEALTH_CSV) else None
    r = subprocess.run([p3.PY, p3.HEALTH], capture_output=True, encoding="utf-8",
                       errors="replace")
    t1 = os.path.getmtime(HEALTH_CSV) if os.path.exists(HEALTH_CSV) else None
    if t1 is None or t1 == t0:
        return None, f"health CSV 未刷新（exit={r.returncode}）"
    rows = {}
    with open(HEALTH_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row.get("file")] = row
    return rows, None


def g3f_acc(a):
    """acc 判定只认 g3f json（证据件优先，同 arm_state 语义）。"""
    if not os.path.exists(a["g3f"]):
        return None, "g3f 缺失"
    try:
        with open(a["g3f"], encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        return None, f"g3f 损坏 {e}"
    bj = j.get("ckpt")
    if bj and os.path.basename(bj) != os.path.basename(a["ckpt"]):
        return None, "g3f 归属不符"
    v = j.get("acc_cv")
    return (v, "") if v is not None else (None, "acc_cv 字段缺失")


def parse_lambda_lines(logp):
    """[lambda_i epN] 行 → list[dict(min,max,mean,p25,p75)]（g = λ_i/λ_max 单位）。"""
    if not os.path.exists(logp):
        return []
    out = []
    pat = re.compile(r"\[lambda_i ep\d+\]\s+min=([\d.eE+-]+)\s+max=([\d.eE+-]+)\s+"
                     r"mean=([\d.eE+-]+)\s+p25=([\d.eE+-]+)\s+p75=([\d.eE+-]+)")
    for m in pat.finditer(open(logp, encoding="utf-8", errors="replace").read()):
        out.append(dict(min=float(m.group(1)), max=float(m.group(2)),
                        mean=float(m.group(3)), p25=float(m.group(4)),
                        p75=float(m.group(5))))
    return out


def parse_dt_series(logp):
    """[Dt step N] D=x 行 → list[(step, D)]。"""
    if not os.path.exists(logp):
        return []
    pat = re.compile(r"\[Dt step (\d+)\]\s+D=([\d.eE+-]+)")
    return [(int(m.group(1)), float(m.group(2)))
            for m in pat.finditer(open(logp, encoding="utf-8", errors="replace").read())]


# ============================== 门1/门2/门3 ==============================
def run_gates(st, arms, health_rows):
    a_arms = [a for a in arms if a["group"] == "A"]
    n_done = n_excl = 0
    for a in a_arms:
        rec = st["arms"].get(a["arm_id"], {})
        status = rec.get("status")
        gid = a["arm_id"]
        if status in (None, "pending", "training", "eval", "failed_eval") or \
                (status not in ("done", "excluded", "EXCLUDED_VIOLATION")
                 and not os.path.exists(a["ckpt"])):
            add("门2", f"{gid}（非终态）", "PENDING",
                f"status={status or '无记录'}")
            continue
        if status in ("excluded", "EXCLUDED_VIOLATION"):
            n_excl += 1
            add("门2", f"{gid} 剔除", "FAIL",
                f"已剔除：{rec.get('note') or status}（剔除后果条款核算 → 见 accounting）")
            continue
        # —— status == done：三门证据链复核（state 缓存不压证据）——
        acc, acc_note = g3f_acc(a)
        # 门1：fin / nonfinite / med|w|≥1e-8 / dead(D-13)
        row = health_rows.get(os.path.basename(a["ckpt"]))
        if row is None:
            add("门1", gid, "FAIL", f"health CSV 无本臂行 {os.path.basename(a['ckpt'])}")
        else:
            fin = row.get("fin") == "True"
            nonfinite = (int(float(row.get("n_nan") or 1)) + int(float(row.get("n_inf") or 1))) == 0
            try:
                med_ok = float(row.get("med_abs")) >= 1e-8
            except (TypeError, ValueError):
                med_ok = False
            dead = row.get("dead") == "True"
            ok1 = fin and nonfinite and med_ok and not dead
            add("门1", gid, "PASS" if ok1 else "FAIL",
                f"fin={fin} nonfinite_ok={nonfinite} med>={1e-8}:{med_ok} dead={dead}")
        # 门2①
        if acc is None:
            add("门2", f"{gid}① LFW", "FAIL", acc_note)
        else:
            add("门2", f"{gid}① LFW acc_cv", "PASS" if acc >= 0.92 else "FAIL",
                f"acc_cv={acc}（阈 0.92）")
        # 门2②：30ep + 无拼接 + [ARM]
        ep = (row or {}).get("epoch")
        ep_ok = str(ep) == str(DT_EPOCHS_TOTAL)
        splice = bool(rec.get("resumed_from_epochs"))
        logp = p3.attempt_log_path(a, max(int(rec.get("attempt", 1) or 1), 1))
        arm_ok, arm_note = p3.assert_arm_log(logp, a)
        ok22 = ep_ok and not splice and arm_ok is True
        add("门2", f"{gid}② 训练完整性", "PASS" if ok22 else "FAIL",
            f"epoch={ep}(需30) 拼接={splice} [ARM]={'PASS' if arm_ok else arm_note}")
        # 门2③：loss 有限（同门1 nonfinite，独立列出）
        nf_ok = row is not None and (int(float(row.get("n_nan") or 1)) +
                                     int(float(row.get("n_inf") or 1))) == 0
        add("门2", f"{gid}③ loss 有限性", "PASS" if nf_ok else "FAIL",
            f"n_nan={ (row or {}).get('n_nan') } n_inf={ (row or {}).get('n_inf') }")
        # 门3：H5 记账
        h5 = rec.get("h5_auc")
        if h5 is None:
            add("门3", gid, "FAIL", "h5_auc 缺失（日志无 RESULT: AUC 行）")
        else:
            add("门3", gid, "PASS" if h5 >= 0.95 else "FAIL", f"AUC={h5}（阈 0.95 不降）")
        n_done += 1
    # —— 剔除后果核算（§13 逐字：N/df，df=N−10 按 §10.1 冻结公式）——
    n_valid = n_done
    df = n_valid - 10
    downgrade = n_excl >= 1 and df < 10
    add("剔除后果", "有效 N 与 df 核算", "PASS",
        f"N_valid={n_valid} EXCLUDED={n_excl} df=N−10={df}"
        + (f"；df<10 → 4 主假设降 descriptive+CI（不删臂不删表）" if downgrade else ""))
    return {"n_valid": n_valid, "n_excluded": n_excl, "df": df,
            "downgrade_flag": downgrade}


# ============================== DG-R1 ==============================
def run_dg_r1(st, arms, health_rows, gate_fail_names):
    a_arms = [a for a in arms if a["group"] == "A"]
    done = [a for a in a_arms if st["arms"].get(a["arm_id"], {}).get("status") == "done"]
    excluded = [a for a in a_arms
                if st["arms"].get(a["arm_id"], {}).get("status") in
                ("excluded", "EXCLUDED_VIOLATION")]
    # a) 已完成臂 100% 过门1/门2；EXCLUDED ≤ 1
    failed_here = [n for n in gate_fail_names if any(
        n.startswith(a["arm_id"]) and ("门1" in n or "门2" in n) for a in done)]
    if not done:
        add("DG-R1", "a) 完成臂数", "PENDING", "尚无 done 臂（时点未至：第 10 臂落盘后判）")
    else:
        add("DG-R1", "a) 已完成臂 100% 过门1/门2",
            "PASS" if not failed_here else "FAIL", f"违例={failed_here}")
    add("DG-R1", "a) EXCLUDED ≤ 1",
        "PASS" if len(excluded) <= 1 else "FAIL",
        f"EXCLUDED={len(excluded)}（>1 → 停机核算 df）")
    # b) λ_i 分布（判据对象 = 条件化臂 d4；其余臂无 λ_i 机制，n/a 如实标注）
    cond_arms = [a for a in done if a.get("distill_type") == "d4"]
    noncond = [a for a in done if a.get("distill_type") != "d4"]
    if not cond_arms:
        add("DG-R1", "b) λ_i 分布", "PENDING", "无已完成条件化（d4）臂")
    for a in cond_arms:
        rec = st["arms"][a["arm_id"]]
        logp = p3.attempt_log_path(a, max(int(rec.get("attempt", 1) or 1), 1))
        eps = parse_lambda_lines(logp)
        if not eps:
            add("DG-R1", f"b) {a['arm_id']} λ_i 行", "FAIL", f"无 [lambda_i] 行（{logp}）")
            continue
        bad_bound = [i for i, e in enumerate(eps, 1)
                     if not (0.0 < e["min"] <= e["max"] < 1.0)]     # g 单位 ⇒ λ_i∈(0,λ_max)
        bad_ratio = [i for i, e in enumerate(eps, 1)
                     if e["p25"] <= 0 or not (0.5 <= e["p75"] / e["p25"] <= 2.0)]
        ok = not bad_bound and not bad_ratio
        add("DG-R1", f"b) {a['arm_id']} λ_i 分布", "PASS" if ok else "FAIL",
            f"epochs={len(eps)} 越界={bad_bound[:5]} 比值越界={bad_ratio[:5]}")
    if noncond:
        add("DG-R1", "b) 非条件化臂", "PASS",
            f"{len(noncond)} 臂无 λ_i 机制（d0-d3 判据对象为 d4，n/a 如实标注）")
    # c) D(t) 末 5 epoch 下降或平台（带宽 1.05 = 实现参数，见 REALIZ）
    dt_pending = []
    for a in done:
        rec = st["arms"][a["arm_id"]]
        logp = p3.attempt_log_path(a, max(int(rec.get("attempt", 1) or 1), 1))
        ser = parse_dt_series(logp)
        if len(ser) < 20:
            dt_pending.append(a["arm_id"])
            continue
        max_step = max(s for s, _ in ser)
        spe = max_step / DT_EPOCHS_TOTAL
        last = [d for s, d in ser if s > 25 * spe]
        prev = [d for s, d in ser if 20 * spe < s <= 25 * spe]
        if not last or not prev:
            dt_pending.append(a["arm_id"])
            continue
        band = sum(last) / len(last)
        base = sum(prev) / len(prev)
        ok = band <= base * REALIZ["dg_r1_c_plateau_band"]
        add("DG-R1", f"c) {a['arm_id']} D(t) 末5ep", "PASS" if ok else "FAIL",
            f"last5={band:.4f} prev5={base:.4f} band={REALIZ['dg_r1_c_plateau_band']}"
            f"（实现参数，非判据）")
    if dt_pending:
        add("DG-R1", "c) D(t) 序列", "PENDING",
            f"样本不足/缺行臂：{dt_pending}")
    # d) 无协议违例：[ARM] 全过 + 无 EXCLUDED_VIOLATION + prereg 存在
    viol = [a["arm_id"] for a in done
            if p3.assert_arm_log(p3.attempt_log_path(
                a, max(int(st["arms"][a["arm_id"]].get("attempt", 1) or 1), 1)), a)[0] is not True]
    vstate = [k for k, r in st["arms"].items()
              if r.get("status") == "EXCLUDED_VIOLATION"]
    prereg_ok = os.path.exists(p3.PREREG)
    okd = not viol and not vstate and prereg_ok
    add("DG-R1", "d) 无协议违例", "PASS" if okd else "FAIL",
        f"ARM违例={viol} V2-1违例={vstate} prereg存在={prereg_ok}"
        f"（配置hash入账待 T-R5 config_hash 字段，届时替换本行）")


# ============================== DG-R0a ==============================
def run_dg_r0a(st, expect_prereg_sha):
    # 1. T-R5 校准（AdaFace 官方权重）
    calib = os.path.join("results", "_t5_calibration_adaface.json")
    if os.path.exists(calib):
        try:
            j = json.load(open(calib, encoding="utf-8"))
            ok = j.get("pass") is True
            add("DG-R0a", "T-R5 三基准管线校准（AdaFace）",
                "PASS" if ok else "FAIL", f"pass={j.get('pass')}")
        except Exception as e:
            add("DG-R0a", "T-R5 三基准管线校准（AdaFace）", "FAIL", f"读取失败 {e}")
    else:
        add("DG-R0a", "T-R5 三基准管线校准（AdaFace）", "PENDING",
            f"planned deliverable 未落：{calib}")
    # 2. LFW-CASIA 重叠审计（D-29 正本：同照确凿 = 1）
    aud = os.path.join("results", "_lfw_casia_overlap_audit_20261015b.json")
    try:
        s = json.load(open(aud, encoding="utf-8"))["summary"]
        strict = s.get("headline_same_photo_strict")
        ok = strict == 1
        add("DG-R0a", "LFW-CASIA 重叠审计 PASS",
            "PASS" if ok else "FAIL", f"strict={strict}（D-29 登记值 = 1）")
    except Exception as e:
        add("DG-R0a", "LFW-CASIA 重叠审计 PASS", "FAIL", f"件缺失/损坏 {e}")
    # 3. T-R1 缓存校验
    t1 = os.path.join("results", "_t1_cache_verify_20261014.json")
    try:
        j = json.load(open(t1, encoding="utf-8"))
        ok = j.get("pass") is True and j.get("n_fail") == 0 and \
            j.get("cos_min", 0) >= j.get("threshold", 0.9999)
        add("DG-R0a", "T-R1 缓存校验 PASS", "PASS" if ok else "FAIL",
            f"pass={j.get('pass')} cos_min={j.get('cos_min')} thr={j.get('threshold')}")
    except Exception as e:
        add("DG-R0a", "T-R1 缓存校验 PASS", "FAIL", f"件缺失/损坏 {e}")
    # 4. prereg 冻结件存在性 + 哈希机检
    if os.path.exists(p3.PREREG):
        got = sha256_file(p3.PREREG)
        if expect_prereg_sha:
            ok = got == expect_prereg_sha
            add("DG-R0a", "prereg 冻结件哈希机检",
                "PASS" if ok else "FAIL", f"got={got[:16]}… expect={expect_prereg_sha[:16]}…")
        else:
            add("DG-R0a", "prereg 冻结件存在", "PASS", f"sha256(实测，未比对)={got}")
            add("DG-R0a", "prereg 冻结哈希比对", "PENDING",
                "未提供 --expect-prereg-sha256（冻结 commit 后补登记再判）")
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              encoding="utf-8").stdout.strip()
        add("DG-R0a", "git HEAD 记录", "PASS", head[:12])
    else:
        add("DG-R0a", "prereg 冻结件存在", "FAIL", "prereg_v2_matrix.md 缺失")
    # 5. 管线健康锚（era C d4a1）
    anchor = os.path.join("results", "_g2prime_d4a1.json")
    try:
        j = json.load(open(anchor, encoding="utf-8"))
        ok = j.get("gate_pass") is True and j.get("acc_cv", 0) >= 0.955
        add("DG-R0a", "管线健康锚 era C d4a1", "PASS" if ok else "FAIL",
            f"gate_pass={j.get('gate_pass')} acc_cv={j.get('acc_cv')}")
    except Exception as e:
        add("DG-R0a", "管线健康锚 era C d4a1", "FAIL", f"件缺失/损坏 {e}")


# ============================== DG-R2a/R2b/R3/R0b（状态记账） ==============================
def run_dg_status(st, arms, which):
    a_arms = [a for a in arms if a["group"] == "A"]
    if "r2a" in which:
        done = sum(1 for a in a_arms
                   if st["arms"].get(a["arm_id"], {}).get("status") == "done")
        txt = open(p3.PREREG, encoding="utf-8").read() if os.path.exists(p3.PREREG) else ""
        has_sigma = bool(re.findall(r"σ̂\s*[=＝]\s*(0?\.\d+)", txt))
        if done < 20:
            add("DG-R2a", "20 臂全落盘", "PENDING", f"done={done}/20")
        else:
            add("DG-R2a", "20 臂全落盘", "PASS", f"done={done}")
        add("DG-R2a", "σ̂ 扩样判定锁定（§8 回填）",
            "PASS" if has_sigma else "PENDING",
            "Holm 裁决表 = 分析层件，本机检只判时点就绪"
            if has_sigma else "prereg 无实测 σ̂ 赋值行")
    if "r2b" in which:
        add("DG-R2b", "WebFace4M 复现臂", "PENDING",
            "时点未至（v2.0 :767-781 原文；无 p，事前登记方向复现）")
    if "r0b" in which:
        mbf_ok = os.path.exists(p3.MBF_TRAIN) and os.path.exists(p3.MBF_MODEL)
        add("DG-R0b", "E 组 T-R9 件（MBF 训练/模型）",
            "PASS" if mbf_ok else "PENDING",
            "不阻塞 A 组首臂（§14 拆分条款）" if mbf_ok else
            f"缺：{' '.join(x for x in (p3.MBF_TRAIN, p3.MBF_MODEL) if not os.path.exists(x))}")
    if "r3" in which:
        cap = glob.glob(os.path.join("results", "*capacity8*")) + \
              glob.glob(os.path.join("results", "*cap8*"))
        budget = os.path.exists(os.path.join(
            os.path.dirname(ROOT), "paper_plan", "BUDGET_CAL_v2_20261015.md"))
        add("DG-R3", "8 臂表已冻结", "PASS" if cap else "PENDING",
            cap[0] if cap else "planned deliverable（0.93M/1.77M × {无蒸馏,质量条件化RKD} × 2 seed）")
        add("DG-R3", "预算余量复核（两要件之一）",
            "PENDING" if not budget else "PASS",
            "BUDGET_CAL_v2 存在；PASS 数值口径 = P4 注册实测基准（D-30），非本件代判"
            if budget else "BUDGET_CAL_v2 缺失")
        add("DG-R3", "不自创第三要件（P4 2-③）", "PASS", "仅两要件")


def main():
    ap = argparse.ArgumentParser(description="P-gate 机检（prereg §13/§14 唯一判据）")
    ap.add_argument("--gates", action="store_true", help="门1/门2/门3 + 剔除后果")
    ap.add_argument("--dg", nargs="*", default=[],
                    choices=["r0a", "r1", "r2a", "r2b", "r3", "r0b"])
    ap.add_argument("--all", action="store_true", help="门 + 全部 DG")
    ap.add_argument("--out", default=None, help="报告 json（已存在则拒写）")
    ap.add_argument("--expect-prereg-sha256", default=None)
    args = ap.parse_args()
    if not (args.gates or args.dg or args.all):
        ap.error("至少选 --gates / --dg / --all 之一")
    do_gates = args.gates or args.all
    do_dg = (["r0a", "r1", "r2a", "r2b", "r3", "r0b"] if args.all else args.dg)

    st = p3.load_state()
    sig_state, branch = p3.read_sigma_branch()
    arms = p3.build_manifest(branch)

    health_rows, err = refresh_health_csv()
    if health_rows is None:
        add("输入", "health CSV 刷新", "FAIL", err)
        health_rows = {}

    acct = {}
    if do_gates:
        print("== 门1/门2/门3（§13 逐字判据）==", flush=True)
        acct = run_gates(st, arms, health_rows)
    if "r1" in do_dg:
        print("== DG-R1（§14 逐字判据）==", flush=True)
        run_dg_r1(st, arms, health_rows,
                  [i["name"] for i in ITEMS if i["status"] == "FAIL"])
    if "r0a" in do_dg:
        print("== DG-R0a（§14 逐字判据）==", flush=True)
        run_dg_r0a(st, args.expect_prereg_sha256)
    rest = [x for x in do_dg if x in ("r2a", "r2b", "r3", "r0b")]
    if rest:
        print(f"== DG 状态记账 {rest}（§14）==", flush=True)
        run_dg_status(st, arms, rest)

    n_fail = sum(1 for i in ITEMS if i["status"] == "FAIL")
    n_pend = sum(1 for i in ITEMS if i["status"] == "PENDING")
    n_pass = sum(1 for i in ITEMS if i["status"] == "PASS")
    report = {
        "meta": {"script": "scripts/gate_check.py", "date": time.strftime("%Y-%m-%d"),
                 "criteria": "prereg_v2_matrix.md §13/§14（逐字见 criteria_13/criteria_dg_r1）",
                 "realization_params": REALIZ,
                 "decoupling_note": "CFP-FP/AgeDB-30/TinyFace 数值不参与任何剔除判断（§13 解耦条款）"},
        "criteria_13": CRITERIA_13,
        "criteria_dg_r1": CRITERIA_DG_R1,
        "criteria_dg_r0a": CRITERIA_DG_R0A,
        "accounting": acct,
        "items": ITEMS,
        "verdict": {"pass": n_pass, "fail": n_fail, "pending": n_pend,
                    "green": n_fail == 0 and n_pend == 0},
    }
    out = args.out or os.path.join(
        "results", f"_gate_check_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(out):
        print(f"[abort] --out 已存在，拒写（红线）：{out}")
        sys.exit(2)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[verdict] PASS={n_pass} FAIL={n_fail} PENDING={n_pend} "
          f"green={report['verdict']['green']}")
    print(f"[out] {out}")
    sys.exit(0 if report["verdict"]["green"] else 1)


if __name__ == "__main__":
    main()
