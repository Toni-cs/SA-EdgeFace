"""小规模训练诊断：300 身份，10 epoch，观察 loss 是否收敛。"""
import os, sys, glob, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.losses import ArcFaceLoss

# 构建 300 身份的小数据集
root = 'datasets/raw/casia-webface'
id_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])[:300]
images, labels = [], []
for label, name in enumerate(id_dirs):
    imgs = sorted(glob.glob(os.path.join(root, name, '*.jpg')))
    for p in imgs:
        img = imread_cn(p)
        if img is None: continue
        images.append(p); labels.append(label)
print(f'小数据集: {len(images)} 图, {len(labels) and labels[-1]+1} 身份')

class MiniDS(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = imread_cn(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112)).astype(np.float32)/127.5 - 1.0
        return torch.from_numpy(np.transpose(img,(2,0,1))), self.labels[idx]

ds = MiniDS(images, labels)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

device = 'cuda'
model = build_edgeface(DEFAULT_CONFIGS['x05_baseline']).to(device)
arc = ArcFaceLoss(512, 300, s=64.0, m=0.5).to(device)
params = list(model.parameters()) + list(arc.parameters())
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-4)
scaler = torch.amp.GradScaler('cuda')

model.train()
for ep in range(10):
    t0 = time.time(); running = 0.0; n = 0
    for imgs, labels_b in dl:
        imgs = imgs.to(device); labels_b = labels_b.to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(imgs)
            loss, _ = arc(emb, labels_b)
        if torch.isnan(loss):
            print('  NaN!'); continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        running += loss.item() * imgs.size(0); n += imgs.size(0)
    print(f'[ep {ep+1}/10] loss={running/max(n,1):.4f} time={time.time()-t0:.0f}s')

# 训练集 Acc
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels_b in dl:
        emb = model(imgs.to(device))
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        w = torch.nn.functional.normalize(arc.weight, p=2, dim=1)
        pred = (emb @ w.T).argmax(1).cpu()
        correct += (pred == labels_b).sum().item(); total += len(labels_b)
print(f'训练集Acc: {correct}/{total} = {correct/total*100:.1f}%')
