import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

t0 = time.time()
CASIA = "datasets/raw/casia-webface"
N = 100

paths, labels = [], []
for label, d in enumerate(sorted(os.listdir(CASIA))[:N]):
    for f in os.listdir(os.path.join(CASIA, d)):
        if f.endswith(".jpg"):
            paths.append(os.path.join(CASIA, d, f))
            labels.append(label)
print(f"[{time.time()-t0:.1f}s] {len(paths)} paths", flush=True)

class DS(Dataset):
    def __init__(self, p, l): self.p, self.l = p, l
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        return torch.from_numpy(np.transpose(np.array(img, np.float32)/127.5-1.0, (2,0,1))), self.l[i], i

ds = DS(paths, labels)

for nw in [0, 2]:
    print(f"\n--- num_workers={nw} ---", flush=True)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, num_workers=nw)
    t1 = time.time()
    for i, (imgs, lbls, idxs) in enumerate(dl):
        if i == 0:
            print(f"  [{time.time()-t1:.1f}s] first batch: {imgs.shape}", flush=True)
        if i >= 3:
            break
    print(f"  [{time.time()-t1:.1f}s] 4 batches done", flush=True)

print(f"\n[{time.time()-t0:.1f}s] ALL DONE", flush=True)