"""
效率评测：参数量、FLOPs、CPU/GPU 推理时延。对比所有配置（含消融）。

用法:
    python scripts/eval_efficiency.py --device cpu --input_size 112 --runs 100
    python scripts/eval_efficiency.py --configs baseline sab_full sab_channel_only
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.edgeface import build_by_name, DEFAULT_CONFIGS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_size=112):
    """轻量 FLOPs 估算：统计 Conv2d/Linear 的 FLOPs。"""
    flops = [0]
    hooks = []

    def conv_hook(m, inp, out):
        bs = out.size(0)
        oh, ow = out.size(2), out.size(3)
        kh, kw = m.kernel_size
        cin = m.in_channels
        cout = m.out_channels
        g = m.groups
        macs = bs * cout * oh * ow * (cin // g) * kh * kw
        flops[0] += 2 * macs

    def linear_hook(m, inp, out):
        bs = inp[0].size(0)
        flops[0] += 2 * bs * m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return flops[0]


def measure_latency(model, device, input_size=112, runs=100, warmup=10):
    model = model.to(device).eval()
    if device == "cpu":
        torch.set_num_threads(1)
    x = torch.randn(1, 3, input_size, input_size).to(device)
    lat = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.time()
            model(x)
            if device != "cpu":
                torch.cuda.synchronize()
            lat.append((time.time() - t0) * 1000)
    lat = np.array(lat)
    return {"mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
            "p95_ms": float(np.percentile(lat, 95)), "all_ms": lat.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Efficiency Evaluation")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--input_size", type=int, default=112)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10, help="预热次数（默认与历史行为一致）")
    parser.add_argument("--repeats", type=int, default=1, help="整段时延测量重复次数（Phase 5 正式表用 3）")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    import platform as _platform
    machine = f"{_platform.machine()} {_platform.processor() or _platform.system()}"

    print(f"\n{'Config':<22} {'Params(M)':>10} {'FLOPs(G)':>10} {'Lat(mean)':>10} {'Lat(med)':>10} {'Lat(P95)':>10}")
    print("-" * 74)
    results = []
    for name in args.configs:
        try:
            model = build_by_name(name)
            params = count_params(model)
            flops = count_flops(model, args.input_size)
            segs = [measure_latency(model, device, args.input_size, args.runs, warmup=args.warmup)
                    for _ in range(max(1, args.repeats))]
            pooled = [x for s0 in segs for x in s0["all_ms"]]
            lat = {"mean_ms": float(np.mean(pooled)), "median_ms": float(np.median(pooled)),
                   "p95_ms": float(np.percentile(pooled, 95)), "all_ms": pooled}
            print(f"{name:<22} {params/1e6:>10.3f} {flops/1e9:>10.4f} "
                  f"{lat['mean_ms']:>10.3f} {lat['median_ms']:>10.3f} {lat['p95_ms']:>10.3f}")
            results.append({"config": name, "params_M": params / 1e6,
                            "flops_G": flops / 1e9, **lat})
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"efficiency_{device}_t{int(time.time())}.txt")
    with open(out_path, "w") as f:
        f.write(f"device={device} input_size={args.input_size} runs={args.runs} warmup={args.warmup} repeats={args.repeats} "
                f"threads={torch.get_num_threads()} machine={machine} time={int(time.time())}\n")
        f.write(f"{'Config':<22} {'Params(M)':>10} {'FLOPs(G)':>10} "
                f"{'Lat_mean':>10} {'Lat_med':>10} {'Lat_p95':>10}\n")
        for r in results:
            f.write(f"{r['config']:<22} {r['params_M']:>10.3f} "
                    f"{r['flops_G']:>10.4f} {r['mean_ms']:>10.3f} "
                    f"{r['median_ms']:>10.3f} {r['p95_ms']:>10.3f}\n")
    print(f"\n[INFO] saved to {out_path}（不覆盖旧文件）")


if __name__ == "__main__":
    main()