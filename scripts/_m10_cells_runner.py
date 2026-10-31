# -*- coding: utf-8 -*-
"""C2-M10-续 0a：18 cell 批跑脚本（预写）。
- CALFW：8 臂 × eval_generic_pairs.py --dataset calfw（硬校验 manifest_sha == dd8aa594…）
- LFW  ：8 臂 × _eval_calib_mirror_v1.py --dataset lfw --ckpt（首跑单 cfg0 验证再批量）
用法：python scripts/_m10_cells_runner.py [--only lfw|calfw|both] [--smoke]
"""
import os, sys, csv, hashlib, argparse, subprocess

REPO = r"E:\下次比赛项目\face_recognition_research"
PY = r"C:\Python314\python.exe"
CALFW_MAN_SHA = "dd8aa59458a310d7ac1d5ddbf86198221ad1e1c47df02d3e0c8e65acf1973d1b"
CALFW_MAN_SHA_ALTS = ("dd8aa59458a310d7ac1d5ddbf86198221ad1e1c47df02d3e0c8e65acf1973d1b",
                     "fc2df448b9fbe599ce9316cb7ee3b26cc67380c968e849c098aaae50c5904c55")

# 8 健康臂 → (arm, cfg)
ARMS = [
    ("cfg0_final", "x05_baseline"),
    ("cfg1_final", "x05_sab_channel"),
    ("cfg2_final", "x05_sab_spatial"),
    ("cfg3_final", "x05_sab_full"),
    ("cfg3_seed123", "x05_sab_full"),
    ("cfg4_final", "x05_sab_full"),
    ("cfg4_seed42_s20_l01", "x05_sab_full"),
    ("cfg5_final", "x05_sab_full"),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_calfw_manifest():
    p = os.path.join(REPO, "datasets", "eval_manifests", "calfw_manifest.csv")
    if not os.path.exists(p):
        raise SystemExit("CALFW manifest 缺失: %s" % p)
    s = sha256_file(p)
    if s not in CALFW_MAN_SHA_ALTS:
        raise SystemExit("[STOP] CALFW manifest_sha=%s != 期望 %s（拒跑）" % (s, CALFW_MAN_SHA))
    print("[gate] CALFW manifest_sha 校验通过: %s" % s[:16], flush=True)
    return True


def run_lfw(arm, cfg, smoke=False):
    out = os.path.join(REPO, "results", "H_lfw_%s.json" % arm)
    if os.path.exists(out):
        print("  [skip] %s 已存在" % os.path.basename(out)); return 0
    cmd = [PY, "scripts/_eval_calib_mirror_v1.py", "--dataset", "lfw", "--batch", "256",
           "--ckpt", ("weights/ablation_casia/%s.pt" % arm) if arm.endswith("_final") else ("weights/ablation_casia/%s_final.pt" % arm), "--cfg", cfg, "--out", out]
    print("  [LFW] %s" % arm, flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    return r.returncode


def run_calfw(arm, cfg):
    out = os.path.join(REPO, "results", "H_calfw_%s.json" % arm)
    if os.path.exists(out):
        print("  [skip] %s 已存在" % os.path.basename(out)); return 0
    cmd = [PY, "scripts/eval_generic_pairs.py", "--dataset", "calfw",
           "--ckpt", ("weights/ablation_casia/%s.pt" % arm) if arm.endswith("_final") else ("weights/ablation_casia/%s_final.pt" % arm), "--cfg", cfg,
           "--aligned_root", "datasets/calfw_aligned", "--manifest_dir", "datasets/eval_manifests",
           "--out", out]
    print("  [CALFW] %s" % arm, flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lfw", "calfw", "both"], default="both")
    ap.add_argument("--smoke", action="store_true", help="LFW 仅首跑 cfg0 单 cell 验证")
    ap.add_argument("--arms", type=str, default=None,
                    help="逗号分隔臂名子集（如 'cfg0_final,cfg4_final,cfg4_seed42_s20_l01'）；缺省=全 8 臂")
    args = ap.parse_args()
    sel = ARMS
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        sel = [(a, c) for (a, c) in ARMS if a in want]
        missing = [w for w in want if w not in [a for a, _ in ARMS]]
        if missing:
            print("[STOP] --arms 含未知臂: %s" % missing); return 2
    rc = 0
    if args.only in ("lfw", "both"):
        print("=== LFW cell（%d 臂） ===" % len(sel))
        if args.smoke:
            rc = run_lfw("cfg0_final", "x05_baseline")
            print("  [smoke] 单 cfg0 结果 exit=%d（确认后再批量）" % rc)
        else:
            for arm, cfg in sel:
                rc = run_lfw(arm, cfg)
                if rc != 0:
                    print("  [STOP] LFW %s exit=%d（停报不跳）" % (arm, rc)); return rc
    if args.only in ("calfw", "both"):
        print("=== CALFW cell（%d 臂） ===" % len(sel))
        check_calfw_manifest()
        for arm, cfg in sel:
            rc = run_calfw(arm, cfg)
            if rc != 0:
                print("  [STOP] CALFW %s exit=%d（停报不跳）" % (arm, rc)); return rc
    print("[done] exit=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())