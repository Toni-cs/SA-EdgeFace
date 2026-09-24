"""极简 CASIA 100id 训练，诊断超时原因。"""
import os, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer

t_start = time.time()
CASIA = "datasets/raw/casia-webface"
N_IDS = 100
EPOCHS = 5
BATCH = 128

print(f"[{time.time()-t_start:.1f}s] building dataset...", flush=True)
paths, labels = [], []
dirs = sorted(os.listdir(CASIA))[:N_IDS]
for label, d in enumerate(dirs):
    for f in os.listdir(os.path.join(CASIA, d)):
        if f.endswith(".jpg"):
            paths.append(os.path.join(CASIA, d, f))
            labels.append(label)
print(f"[{time.time()-t_start:.1f}s] {len(paths)} imgs, {N_IDS} ids", flush=True)

class DS(Dataset):
    def __init__(self, p, l):
        self.p, self.l = p, l
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        img = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2,0,1))), self.l[i], i

ds = DS(paths, labels)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
print(f"[{time.time()-t_start:.1f}s] dataloader ready", flush=True)

trainer = DistillTrainer(DEFAULT_CONFIGS["x05_baseline"], N_IDS, device="cuda:0",
                         w_arc=1.0, w_feat=0.0, w_rel=0.0)
print(f"[{time.time()-t_start:.1f}s] trainer ready, starting train...", flush=True)

trainer.train(dl, None, epochs=EPOCHS, lr=1e-3,
              save_dir="weights/casia_test", save_name="test",
              log_every=10, use_amp=True)
print(f"[{time.time()-t_start:.1f}s] DONE", flush=True)