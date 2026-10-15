"""T-R4 统计件：FLOPs 与实际延迟对比表（白皮书卡 :600-602 逐字落码，prereg pin 1 组件）。

卡原文：
  5. FLOPs 与实际延迟统计：thop/torchinfo 输出 FLOPs 对比表；
     torch.cuda.Event 计时 GPU 延迟；CPU 延迟用 time.perf_counter
     （batch=1，warmup 50 + 测量 200 次，报 p50/p95）
  人工必查点（:604-605）：延迟测量方法学（warmup、batch=1、多次统计）——
     AI 生成的延迟代码常见缺陷是无 warmup 导致首帧开销污染均值。
  检查清单（:700）：T-R4：延迟测量含 warmup、多次统计、batch=1。

落码口径（全部字段写入输出 JSON，论文数字可溯源 [:718]）：
  - FLOPs = thop（JSON 记 thop 版本；thop 计 MAC，flops := 2×macs，两字段同报披露口径）
  - GPU 延迟 = torch.cuda.Event 成对逐次 elapsed_time（ms）；CPU 延迟 = time.perf_counter（ms）
  - batch=1 固定；warmup=50、测量 200 次为默认（可 flag 覆盖用于冒烟，JSON 记实际值）
  - 报 p50/p95（附 mean/n）；对比维度 = config × backbone（双骨干）
  - 推理路径 = get_embedding（与评测同径，含 L2 归一化）
  - --out 已存在拒绝覆盖（exit 5）

用法：
  python scripts/stats_efficiency.py --out results/_t4_efficiency_20261015.json
  冒烟：python scripts/stats_efficiency.py --configs cfg4 --backbones pconv \
            --warmup 3 --runs 10 --out results/_t4_smoke_xxx.json
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.edgeface import build_edgeface, DEFAULT_CONFIGS  # noqa: E402

GPU_METHOD = "torch.cuda.Event(pair, per-iter, elapsed_time ms)"
CPU_METHOD = "time.perf_counter(per-iter ms)"


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def flops_thop(model, input_size=112):
    """thop 口径 FLOPs（卡 :600 指定方法）。返回 macs/params/flops(=2×macs)/版本。"""
    import thop
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        macs, params = thop.profile(model, inputs=(x,), verbose=False)
    return {
        "macs": float(macs),
        "params": float(params),
        "flops": float(macs) * 2.0,
        "method": "thop",
        "thop_version": str(getattr(thop, "__version__", "unknown")),
        "flops_convention": "thop 计 MAC；flops = 2*macs（两字段同报）",
    }


def summarize(times_ms):
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": int(len(arr)),
    }


def measure_gpu(model, x, runs, warmup):
    """卡 :601：torch.cuda.Event 计时；warmup 先行，逐次成对事件。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    return summarize(times)


def measure_cpu(model, x, runs, warmup):
    """卡 :601：CPU 用 time.perf_counter；warmup 先行，逐次计时。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return summarize(times)


def build_model(cfg, backbone):
    model = build_edgeface(DEFAULT_CONFIGS[cfg], backbone_type=backbone)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="T-R4 效率统计（FLOPs + GPU/CPU 延迟）")
    ap.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS.keys()))
    ap.add_argument("--backbones", nargs="+", default=["shufflenet", "pconv"])
    ap.add_argument("--warmup", type=int, default=50, help="卡口径默认 50")
    ap.add_argument("--runs", type=int, default=200, help="卡口径默认 200")
    ap.add_argument("--input_size", type=int, default=112)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_gpu", action="store_true", help="跳过 GPU 延迟段")
    args = ap.parse_args()

    out_path = args.out or (
        f"results/_t4_efficiency_{datetime.date.today().isoformat()}.json")
    if os.path.exists(out_path):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}（请传 --out <新路径>）")
        sys.exit(5)

    cuda_ok = torch.cuda.is_available() and not args.no_gpu
    print(f"[meta] batch=1 warmup={args.warmup} runs={args.runs} "
          f"gpu={'on' if cuda_ok else 'off'} threads={torch.get_num_threads()}")

    rows = []
    header = (f"{'config':<8} {'backbone':<11} {'params_M':>9} {'flops_G':>9} "
              f"{'gpu_p50':>9} {'gpu_p95':>9} {'cpu_p50':>9} {'cpu_p95':>9}")
    print(header)
    print("-" * len(header))
    thop_meta = None
    for cfg in args.configs:
        for backbone in args.backbones:
            model = build_model(cfg, backbone)
            fl = flops_thop(model, args.input_size)
            thop_meta = {"version": fl["thop_version"], "flops_convention": fl["flops_convention"]}
            params = count_params(model)
            row = {
                "config": cfg,
                "backbone": backbone,
                "params_M": params / 1e6,
                "params": params,
                "macs_G": fl["macs"] / 1e9,
                "flops_G": fl["flops"] / 1e9,
                "gpu_ms": None,
                "cpu_ms": None,
            }
            if cuda_ok:
                model = model.cuda()
                xg = torch.randn(1, 3, args.input_size, args.input_size, device="cuda")
                row["gpu_ms"] = measure_gpu(model, xg, args.runs, args.warmup)
                del xg
            model = model.cpu()
            xc = torch.randn(1, 3, args.input_size, args.input_size)
            row["cpu_ms"] = measure_cpu(model, xc, args.runs, args.warmup)
            del xc, model

            g50 = row["gpu_ms"]["p50"] if row["gpu_ms"] else float("nan")
            g95 = row["gpu_ms"]["p95"] if row["gpu_ms"] else float("nan")
            print(f"{cfg:<8} {backbone:<11} {row['params_M']:>9.3f} {row['flops_G']:>9.4f} "
                  f"{g50:>9.3f} {g95:>9.3f} {row['cpu_ms']['p50']:>9.3f} "
                  f"{row['cpu_ms']['p95']:>9.3f}")
            rows.append(row)

    payload = {
        "schema": "t4_efficiency_v1",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/stats_efficiency.py",
        "machine": f"{platform.machine()} {platform.processor() or platform.system()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "thop": thop_meta,
        "batch": 1,
        "warmup": args.warmup,
        "runs": args.runs,
        "input_size": args.input_size,
        "cpu_threads": torch.get_num_threads(),
        "gpu_measured": bool(cuda_ok),
        "timing": {"gpu": GPU_METHOD, "cpu": CPU_METHOD},
        "inference_path": "get_embedding (forward + L2 normalize)，与评测同径",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        print(f"[ERROR] 目标文件已存在，拒绝覆盖: {out_path}")
        sys.exit(5)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(rows)} 行 → {out_path}")


if __name__ == "__main__":
    main()
