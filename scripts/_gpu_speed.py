"""测 GPU bs=256 速度 + 估算全量训练时间。"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface

device = "cuda:0"
model = build_edgeface(DEFAULT_CONFIGS["x05_sab_full"]).to(device)
x = torch.randn(256, 3, 112, 112).to(device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    y = model(x)
    loss = y.sum()
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()
elapsed = time.time() - t0
ms = elapsed / 20 * 1000
print(f"bs=256: {ms:.1f}ms/iter")
steps = 490623 // 256
print(f"full CASIA: {steps} steps/epoch, {steps*ms/1000/60:.1f} min/epoch")
for ep in [5, 10, 20, 30]:
    print(f"  {ep}ep × 6cfg: {steps*ms/1000/60*ep*6/60:.1f} hours")