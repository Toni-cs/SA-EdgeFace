"""CASIA 100id GPU 训练单配置。用法: python scripts/_casia100.py <0-5>"""
import os, sys, time, glob
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer
from qad_face.data.surveillance_aug import SurveillanceAugment
from sklearn.metrics import roc_auc_score

CASIA = "datasets/raw/casia-webface"
N_IDS = 50; EPOCHS = 10; BATCH = 128
SAVE = "weights/casia50"

CONFIGS = [
    ("x05_baseline",0.0,False,"baseline"),
    ("x05_sab_channel",0.0,False,"+C1 channel"),
    ("x05_sab_spatial",0.0,False,"+C1 spatial"),
    ("x05_sab_full",0.0,False,"+C1 full SAB"),
    ("x05_sab_full",0.1,False,"+C1+C2 distill"),
    ("x05_sab_full",0.1,True,"+C1+C2+C3 aug"),
]

class DS(Dataset):
    def __init__(self, root, n_ids, use_aug=False):
        self.paths, self.labels = [], []
        self.aug = SurveillanceAugment() if use_aug else None
        for l, d in enumerate(sorted(os.listdir(root))[:n_ids]):
            for f in os.listdir(os.path.join(root, d)):
                if f.endswith(".jpg"):
                    self.paths.append(os.path.join(root, d, f)); self.labels.append(l)
        self.nc = n_ids
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = np.array(img, dtype=np.uint8)
        if self.aug and np.random.random()<0.5: img = self.aug(img)
        img = img.astype(np.float32)/127.5-1.0
        return torch.from_numpy(np.transpose(img,(2,0,1))), self.labels[i], i

def teacher(ds):
    c = os.path.join(SAVE, "teacher50.npy")
    if os.path.exists(c): return torch.from_numpy(np.load(c)).float()
    import onnxruntime as ort
    s = ort.InferenceSession("weights/models/buffalo_s/w600k_mbf.onnx", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    i_, o_ = s.get_inputs()[0].name, s.get_outputs()[0].name
    f = np.zeros((len(ds),512), np.float32)
    for i in range(0, len(ds), 256):
        b = np.zeros((min(256,len(ds)-i),3,112,112), np.float32)
        for j in range(len(b)):
            img = Image.open(ds.paths[i+j]).convert("RGB")
            b[j] = np.transpose(np.array(img,np.float32)/127.5-1.0,(2,0,1))
        f[i:i+len(b)] = s.run([o_],{i_:b})[0][:,:512]
    os.makedirs(SAVE, exist_ok=True); np.save(c, f)
    return torch.from_numpy(f).float()

def ev(model, ds):
    model.eval()
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    fs, ls = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            fs.append(model.get_embedding(x.cuda()).cpu().numpy()); ls.append(y.numpy())
    fs = np.concatenate(fs); ls = np.concatenate(ls)
    n = np.linalg.norm(fs, axis=1)+1e-8
    ss, sa = [], []
    for i in range(0, len(fs), 5):
        for j in range(i+1, min(i+50, len(fs))):
            ss.append(np.dot(fs[i],fs[j])/(n[i]*n[j])); sa.append(1 if ls[i]==ls[j] else 0)
    ss, sa = np.array(ss), np.array(sa)
    if len(set(sa))<2: return 0,0,0
    auc = roc_auc_score(sa, ss)
    acc = max(np.mean((ss>=t).astype(int)==sa) for t in np.linspace(ss.min(),ss.max(),200))
    return auc, acc, ss[sa==1].mean()-ss[sa==0].mean()

idx = int(sys.argv[1])
cfg, wf, aug, desc = CONFIGS[idx]
t0 = time.time()
ds = DS(CASIA, N_IDS, use_aug=aug)
print(f"[{desc}] {len(ds)} imgs, {ds.nc} ids", flush=True)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
tf = teacher(ds) if wf > 0 else None
tr = DistillTrainer(DEFAULT_CONFIGS[cfg], ds.nc, device="cuda:0", w_feat=wf, w_rel=0.0)
tr.train(dl, tf, epochs=EPOCHS, lr=1e-3, save_dir=SAVE, save_name=f"cfg{idx}", log_every=20, use_amp=True)
m = build_edgeface(DEFAULT_CONFIGS[cfg]).cuda()
m.load_state_dict(torch.load(f"{SAVE}/cfg{idx}_final.pt")["student"], strict=False)
auc, acc, sep = ev(m, ds)
print(f"\n  AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} ({time.time()-t0:.0f}s)", flush=True)
with open(os.path.join(SAVE, "results.txt"), "a") as f:
    f.write(f"{desc}\t{auc:.4f}\t{acc*100:.2f}\t{sep:.4f}\n")