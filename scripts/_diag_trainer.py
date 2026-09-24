import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t0 = time.time()
import torch
from qad_face.train.distill import DistillTrainer
from qad_face.models.edgeface import DEFAULT_CONFIGS
print(f"[{time.time()-t0:.1f}s] imports done", flush=True)

trainer = DistillTrainer(DEFAULT_CONFIGS["x05_baseline"], 10, device="cuda:0",
                         w_arc=1.0, w_feat=0.0, w_rel=0.0)
print(f"[{time.time()-t0:.1f}s] trainer created", flush=True)

# 简单训练 1 step
from torch.utils.data import DataLoader
x = torch.randn(10, 3, 112, 112)
labels = torch.arange(10)
idxs = torch.arange(10)
print(f"[{time.time()-t0:.1f}s] test input ready", flush=True)

trainer.student.train()
trainer.student.zero_grad()
emb = trainer.student(x.cuda())
print(f"[{time.time()-t0:.1f}s] forward: {emb.shape}", flush=True)
loss, _, logs = trainer.criterion(emb, None, labels.cuda())
print(f"[{time.time()-t0:.1f}s] loss: {loss.item():.4f}", flush=True)
loss.backward()
print(f"[{time.time()-t0:.1f}s] backward OK", flush=True)
print(f"[{time.time()-t0:.1f}s] ALL DONE", flush=True)