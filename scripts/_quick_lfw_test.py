"""快速验证 LFW 评测 pipeline：10 对，CPU，直接内联。"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface

t0 = time.time()
model = build_edgeface(DEFAULT_CONFIGS["edgeface_x0_5_sab"])
ckpt = torch.load("weights/distill_mini/mini_x05_final.pt", map_location="cpu")
model.load_state_dict(ckpt["student"], strict=False)
model.eval()
print(f"model loaded: {time.time()-t0:.1f}s")

lfw_dir = "datasets/lfw"
with open("datasets/pairs_mini.txt") as f:
    lines = f.readlines()[1:11]

pairs = []
for line in lines:
    parts = line.strip().split()
    if len(parts) == 3:
        pairs.append((parts[0], int(parts[1]), parts[0], int(parts[2]), True))
    elif len(parts) == 4:
        pairs.append((parts[0], int(parts[1]), parts[2], int(parts[3]), False))

print(f"pairs: {len(pairs)}")

def extract(name, idx):
    p = os.path.join(lfw_dir, name, f"{name}_{idx:04d}.jpg")
    if not os.path.exists(p):
        return np.zeros(512, dtype=np.float32)
    img = imread_cn(p)
    if img is None:
        return np.zeros(512, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        return model.get_embedding(t).numpy().flatten()

sims, labels = [], []
for n1, i1, n2, i2, same in pairs:
    f1 = extract(n1, i1)
    f2 = extract(n2, i2)
    sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
    sims.append(sim)
    labels.append(1 if same else 0)
    print(f"  {n1}_{i1} vs {n2}_{i2}: sim={sim:.4f} same={same}")

from sklearn.metrics import roc_auc_score
sims, labels = np.array(sims), np.array(labels)
auc = roc_auc_score(labels, sims) if len(set(labels)) > 1 else float("nan")
best_acc = max(np.mean((sims >= t).astype(int) == labels) for t in np.linspace(sims.min(), sims.max(), 100))
print(f"\nAUC={auc:.4f} Acc={best_acc*100:.1f}%")
print(f"[DONE] {time.time()-t0:.1f}s")