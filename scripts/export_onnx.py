"""
导出训练好的 EdgeFace 为 ONNX（用于部署和 ONNX 评测）。

用法:
    python scripts/export_onnx.py --ckpt weights/distill/student_final.pt \
        --config sab_full --output weights/distill/student_sab_full.onnx
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.edgeface import build_edgeface, DEFAULT_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Export EdgeFace to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="训练 checkpoint 路径")
    parser.add_argument("--config", type=str, default="sab_full",
                        choices=list(DEFAULT_CONFIGS.keys()))
    parser.add_argument("--output", type=str, required=True, help="输出 ONNX 路径")
    parser.add_argument("--input_size", type=int, default=112)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    model = build_edgeface(DEFAULT_CONFIGS[args.config])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["student"] if "student" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"], output_names=["embedding"],
        opset_version=args.opset, dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"[OK] exported to {args.output}")


if __name__ == "__main__":
    main()