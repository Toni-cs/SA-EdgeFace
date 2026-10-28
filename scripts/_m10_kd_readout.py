# -*- coding: utf-8 -*-
"""C2-M10-续 0b：KD 四 Δ 判读模板（预写）。
四 Δ = (cfg4−cfg0)@{LFW,CALFW} + (cfg4_s20_l01−cfg0)@{LFW,CALFW}
判读（v3.6 冻结条款）：四 Δ 全正 → 叙事句「域内保持/跨域受损」成立；否则不成立（仅报数）。
读出源：results/H_lfw_<arm>.json（_eval_calib_mirror_v1 输出 value=acc%）；
        results/H_calfw_<arm>.json（eval_generic_pairs 输出 value=acc_cv10）。
"""
import os, json, sys

REPO = r"E:\下次比赛项目\face_recognition_research"
KD_ARMS = ["cfg4_final", "cfg4_seed42_s20_l01"]
BASE_ARM = "cfg0_final"


def load(proto, arm):
    p = os.path.join(REPO, "results", "H_%s_%s.json" % (proto, arm))
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return float(d["value"])


def main():
    print("=== KD 四 Δ 判读（v3.6 冻结条款）===")
    base = {proto: load(proto, BASE_ARM) for proto in ("lfw", "calfw")}
    if any(v is None for v in base.values()):
        print("[WAIT] cfg0 两列未齐（%s）→ 待 18 cell 出数" % base); return 2
    print("cfg0: LFW=%.4f CALFW=%.4f" % (base["lfw"], base["calfw"]))
    deltas = []
    for arm in KD_ARMS:
        for proto in ("lfw", "calfw"):
            v = load(proto, arm)
            if v is None:
                print("[WAIT] %s@%s 未出" % (arm, proto)); return 2
            d = v - base[proto]
            deltas.append((arm, proto, v, d))
            print("  Δ(%s@%s) = %.4f − %.4f = %+.4f" % (arm, proto, v, base[proto], d))
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for _, _, _, d in deltas]
    print("---")
    if all(s == 1 for s in signs):
        print("判读：**四 Δ 全正** → 「域内保持/跨域受损」叙事句**成立**（描述级；主张级仍需 +1.0pp）")
    elif all(s == -1 for s in signs):
        print("判读：**四 Δ 全负** → 叙事句**不成立**（数据照报不删）")
    else:
        print("判读：**混合**（符号 %s）→ 叙事句**不成立**，仅报数（数字不删不改）" % signs)
    return 0


if __name__ == "__main__":
    sys.exit(main())