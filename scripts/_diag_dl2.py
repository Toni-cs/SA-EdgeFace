import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

t0 = time.time()
CASIA = "datasets/raw/casia-webface"
paths, labels = [], []
for label, d in enumerate(sorted(os.listdir(CASIA))[:20]):
    for f in os.listdir(os.path.join(CASIA, d)):
        if f.endswith(".jpg"):
            paths.append(os.path.join(CASIA, d, f))
            labels.append(label)
print(f"[{time.time()-t0:.1f}s] {len(paths)} paths", flush=True)

class DS(Dataset):
    def __len__(self): return len(paths)
    def __getitem__(self, i):
        img = Image.open(paths[i]).convert("RGB")
        return torch.from_numpy(np.transpose(np.array(img, np.float32)/127.5-1.0, (2,0,1))), labels[i], i

ds = DS()
dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
print(f"[{time.time()-t0:.1f}s] dataloader ready", flush=True)

t1 = time.time()
for i, batch in enumerate(dl):
    if i % 5 == 0:
        print(f"  [{time.time()-t1:.1f}s] batch {i}: {batch[0].shape}", flush=True)
    if i >= 14:
        break
print(f"[{time.time()-t0:.1f}s] DONE", flush=True)