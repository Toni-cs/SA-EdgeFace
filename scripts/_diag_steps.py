import time
t0 = time.time()
print(f"[{time.time()-t0:.1f}s] start", flush=True)

import torch
print(f"[{time.time()-t0:.1f}s] torch imported", flush=True)

print(f"[{time.time()-t0:.1f}s] checking cuda...", flush=True)
print(f"  available: {torch.cuda.is_available()}", flush=True)
print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)

x = torch.randn(4, 3, 112, 112).cuda()
print(f"[{time.time()-t0:.1f}s] cuda tensor OK", flush=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
print(f"[{time.time()-t0:.1f}s] edgeface imported", flush=True)

model = build_edgeface(DEFAULT_CONFIGS["x05_baseline"]).cuda()
print(f"[{time.time()-t0:.1f}s] model on GPU", flush=True)

y = model(x)
print(f"[{time.time()-t0:.1f}s] forward OK: {y.shape}", flush=True)

loss = y.sum()
loss.backward()
print(f"[{time.time()-t0:.1f}s] backward OK", flush=True)

from qad_face.train.distill import DistillTrainer
print(f"[{time.time()-t0:.1f}s] DistillTrainer imported", flush=True)

trainer = DistillTrainer(DEFAULT_CONFIGS["x05_baseline"], 10, device="cuda:0",
                         w_arc=1.0, w_feat=0.0, w_rel=0.0)
print(f"[{time.time()-t0:.1f}s] trainer created", flush=True)

print(f"[{time.time()-t0:.1f}s] ALL OK", flush=True)