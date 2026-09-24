"""
模型代码冒烟验证：检查所有配置可构建、前向 shape 正确、loss/增强可运行。
不依赖 insightface 或大数据集，仅验证 qad_face 模型/训练代码本身。

用法: python scripts/verify_models.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.edgeface import build_by_name, build_edgeface, DEFAULT_CONFIGS
from qad_face.models.attention import SurveillanceAttentionBlock
from qad_face.data.surveillance_aug import SurveillanceAugment
from qad_face.train.losses import ArcFaceLoss, DistillTotalLoss


def check(cond, msg):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    print("== 1. 模型构建与前向 ==")
    x = torch.randn(2, 3, 112, 112)
    for name in DEFAULT_CONFIGS:
        m = build_by_name(name)
        with torch.no_grad():
            out = m(x)
        check(out.shape == (2, 512), f"{name}: output shape {tuple(out.shape)} == (2,512)")
    print("  all configs forward OK")

    print("\n== 2. SAB 消融分支开关 ==")
    for cfg in [
        {"use_channel": True, "use_spatial": False, "use_detail": False},
        {"use_channel": False, "use_spatial": True, "use_detail": False},
        {"use_channel": False, "use_spatial": False, "use_detail": True},
        {"use_channel": True, "use_spatial": True, "use_detail": True},
    ]:
        sab = SurveillanceAttentionBlock(64, **cfg)
        with torch.no_grad():
            y = sab(torch.randn(2, 64, 28, 28))
        check(y.shape == (2, 64, 28, 28), f"SAB cfg {cfg} shape {tuple(y.shape)}")

    print("\n== 3. C3 监控退化增强 ==")
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    for sev in ["none", "light", "medium", "heavy"]:
        aug = SurveillanceAugment(severity=sev)
        out = aug(img.copy())
        check(out.shape == (112, 112, 3), f"aug severity={sev} shape {out.shape}")

    print("\n== 4. ArcFace + 蒸馏 loss ==")
    emb_dim, num_classes = 512, 100
    arc = ArcFaceLoss(emb_dim, num_classes)
    emb = torch.randn(8, emb_dim, requires_grad=True)
    label = torch.randint(0, num_classes, (8,))
    loss_arc, _ = arc(emb, label)
    check(loss_arc.item() > 0, f"ArcFace loss={loss_arc.item():.4f} > 0")

    crit = DistillTotalLoss(arc, w_arc=1.0, w_feat=0.5, w_rel=0.1)
    t_emb = torch.randn(8, emb_dim)
    total, _, logs = crit(emb, t_emb, label)
    check(total.item() > 0, f"Distill total={total.item():.4f} > 0, logs={ {k: round(v,3) for k,v in logs.items()} }")

    crit_no_distill = DistillTotalLoss(arc, w_arc=1.0, w_feat=0.0, w_rel=0.0)
    total2, _, _ = crit_no_distill(emb, None, label)
    check(total2.item() > 0, f"no-distill total={total2.item():.4f} > 0")

    print("\n== 5. 参数量对比（baseline vs sab_full） ==")
    for name in ["baseline", "sab_full", "edgeface_x0_5", "edgeface_x0_5_sab"]:
        m = build_by_name(name)
        p = sum(pp.numel() for pp in m.parameters())
        print(f"  {name}: {p/1e6:.3f} M")

    print("\n== 6. 梯度回传 ==")
    m = build_by_name("sab_full")
    m.train()
    out = m(x)
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    check(has_grad, "backward produces non-zero gradients")

    print("\n[ALL PASSED] 模型代码冒烟验证通过。")


if __name__ == "__main__":
    main()