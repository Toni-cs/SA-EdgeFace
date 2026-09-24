"""测试 EdgeFace 在 GPU 上能否完整前向+反向。"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface

device = "cuda:0"
model = build_edgeface(DEFAULT_CONFIGS["x05_sab_full"]).to(device)
x = torch.randn(4, 3, 112, 112).to(device)
try:
    y = model(x)
    print(f"forward OK: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("backward OK")
    # 测速
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        y = model(x)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"30 iters: {elapsed:.2f}s = {elapsed/30*1000:.1f}ms/iter (bs=4)")
    # bs=64
    x64 = torch.randn(64, 3, 112, 112).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = model(x64)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"10 iters bs=64: {elapsed:.2f}s = {elapsed/10*1000:.1f}ms/iter")
    print(f"est. 490k imgs, bs64: {490623//64 * elapsed/10 / 60:.0f} min/epoch")
    print(f"est. 6×30ep: {490623//64 * elapsed/10 * 180 / 3600:.1f} hours")
except Exception as e:
    print(f"FAIL: {e}")