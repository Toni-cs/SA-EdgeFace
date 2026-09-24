"""极小规模测速：10 身份 1 epoch。"""
import os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from qad_face.models.edgeface import DEFAULT_CONFIGS
from qad_face.train.distill import DistillTrainer

CASIA = "datasets/raw/casia-webface"
ds_imgs, ds_labels = [], []
dirs = sorted(os.listdir(CASIA))[:10]
for label, d in enumerate(dirs):
    for f in sorted(os.listdir(os.path.join(CASIA, d)))[:10]:
        img = Image.open(os.path.join(CASIA, d, f)).convert("RGB")
        ds_imgs.append(torch.from_numpy(np.transpose(np.array(img, dtype=np.float32)/127.5-1.0, (2,0,1))))
        ds_labels.append(label)

class MiniDS(torch.utils.data.Dataset):
    def __len__(self): return len(ds_imgs)
    def __getitem__(self, i): return ds_imgs[i], ds_labels[i], i

ds = MiniDS()
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
trainer = DistillTrainer(DEFAULT_CONFIGS["x05_sab_full"], 10, device="cpu", w_feat=0.0, w_rel=0.0)
t0 = time.time()
trainer.train(dl, None, epochs=1, lr=1e-3, save_dir="weights/st", save_name="s", log_every=5, use_amp=False)
elapsed = time.time() - t0
n_steps = len(dl)
print(f"\n{n_steps} steps in {elapsed:.1f}s = {elapsed/n_steps:.2f}s/step")
print(f"full CASIA 490k imgs, bs64: {490623//64} steps/ep")
print(f"  1 epoch: {490623//64 * elapsed/n_steps/60:.0f} min")
print(f"  6×30ep: {490623//64 * elapsed/n_steps*180/3600:.0f} hours")
