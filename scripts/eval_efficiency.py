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
    x = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    return elapsed / runs * 1000


def main():
    parser = argparse.ArgumentParser(description="Efficiency Evaluation")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--input_size", type=int, default=112)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print(f"\n{'Config':<22} {'Params(M)':>10} {'FLOPs(G)':>10} {'Latency(ms)':>12}")
    print("-" * 58)
    results = []
    for name in args.configs:
        try:
            model = build_by_name(name)
            params = count_params(model)
            flops = count_flops(model, args.input_size)
            lat = measure_latency(model, device, args.input_size, args.runs)
            print(f"{name:<22} {params/1e6:>10.3f} {flops/1e9:>10.4f} {lat:>12.3f}")
            results.append({"config": name, "params_M": params / 1e6,
                            "flops_G": flops / 1e9, "latency_ms": lat})
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "efficiency.txt")
    with open(out_path, "w") as f:
        f.write(f"device={device} input_size={args.input_size} runs={args.runs}\n")
        f.write(f"{'Config':<22} {'Params(M)':>10} {'FLOPs(G)':>10} {'Latency(ms)':>12}\n")
        for r in results:
            f.write(f"{r['config']:<22} {r['params_M']:>10.3f} "
                    f"{r['flops_G']:>10.4f} {r['latency_ms']:>12.3f}\n")
    print(f"\n[INFO] saved to {out_path}")


if __name__ == "__main__":
    main()