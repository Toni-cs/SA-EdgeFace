"""CASIA 500id 子集训练，单配置，快速完成。"""
import os, sys, time, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer
from qad_face.data.surveillance_aug import SurveillanceAugment
from sklearn.metrics import roc_auc_score

CASIA = "datasets/raw/casia-webface"
DEVICE = "cuda:0"
EPOCHS = 15
BATCH = 256
N_IDS = 500
SAVE_DIR = "weights/ablation_casia500"
TEACHER_CACHE = os.path.join(SAVE_DIR, f"teacher_{N_IDS}.npy")

CONFIGS = [
    ("x05_baseline",    0.0, False, "baseline"),
    ("x05_sab_channel", 0.0, False, "+C1 channel"),
    ("x05_sab_spatial", 0.0, False, "+C1 spatial"),
    ("x05_sab_full",    0.0, False, "+C1 full SAB"),
    ("x05_sab_full",    0.1, False, "+C1+C2 distill"),
    ("x05_sab_full",    0.1, True,  "+C1+C2+C3 aug"),
]

class CasiaSubset(Dataset):
    def __init__(self, root, n_ids, use_aug=False):
        self.paths, self.labels = [], []
        self.use_aug = use_aug
        self.aug = SurveillanceAugment() if use_aug else None
        dirs = sorted(os.listdir(root))[:n_ids]
        for label, d in enumerate(dirs):
            for f in os.listdir(os.path.join(root, d)):
                if f.endswith(".jpg"):
                    self.paths.append(os.path.join(root, d, f))
                    self.labels.append(label)
        self.num_classes = n_ids
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img, dtype=np.uint8)
        if self.use_aug and np.random.random() < 0.5:
            img = self.aug(img)
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2,0,1))), self.labels[idx], idx

def precompute_teacher(ds):
    if os.path.exists(TEACHER_CACHE):
        return torch.from_numpy(np.load(TEACHER_CACHE)).float()
    import onnxruntime as ort
    print(f"[Teacher] precomputing {len(ds)} imgs...", flush=True)
    sess = ort.InferenceSession("weights/models/buffalo_s/w600k_mbf.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    feats = np.zeros((len(ds), 512), dtype=np.float32)
    bs = 256
    for i in range(0, len(ds), bs):
        batch = np.zeros((min(bs, len(ds)-i), 3, 112, 112), dtype=np.float32)
        for j in range(len(batch)):
            img = Image.open(ds.paths[i+j]).convert("RGB")
            batch[j] = np.transpose(np.array(img, dtype=np.float32)/127.5-1.0, (2,0,1))
        feats[i:i+len(batch)] = sess.run([out], {inp: batch})[0][:, :512]
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TEACHER_CACHE, feats)
    print(f"  done: {feats.shape}", flush=True)
    return torch.from_numpy(feats).float()

def eval_model(model, ds, device):
    model.eval()
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in dl:
            feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
            labels.append(lbls.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    norms = np.linalg.norm(feats, axis=1) + 1e-8
    sims, same = [], []
    for i in range(0, len(feats), 5):
        for j in range(i+1, min(i+50, len(feats))):
            sims.append(np.dot(feats[i], feats[j]) / (norms[i]*norms[j]))
            same.append(1 if labels[i]==labels[j] else 0)
    sims, same = np.array(sims), np.array(same)
    if len(set(same)) < 2: return 0, 0, 0
    auc = roc_auc_score(same, sims)
    acc = max(np.mean((sims>=t).astype(int)==same) for t in np.linspace(sims.min(), sims.max(), 200))
    sep = sims[same==1].mean() - sims[same==0].mean()
    return auc, acc, sep

idx = int(sys.argv[1])
cfg_name, w_feat, use_aug, desc = CONFIGS[idx]
print(f"[{idx+1}/6] {desc} ({cfg_name}) w_feat={w_feat} aug={use_aug}", flush=True)
ds = CasiaSubset(CASIA, N_IDS, use_aug=use_aug)
print(f"  {len(ds)} imgs, {ds.num_classes} ids", flush=True)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
tf = precompute_teacher(ds) if w_feat > 0 else None
trainer = DistillTrainer(DEFAULT_CONFIGS[cfg_name], ds.num_classes, device=DEVICE,
                         w_arc=1.0, w_feat=w_feat, w_rel=0.0, feat_mode="cosine")
t0 = time.time()
trainer.train(dl, tf, epochs=EPOCHS, lr=1e-3,
              save_dir=SAVE_DIR, save_name=f"cfg{idx}", log_every=50, use_amp=True)
model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(DEVICE)
ckpt = torch.load(f"{SAVE_DIR}/cfg{idx}_final.pt", map_location=DEVICE)
model.load_state_dict(ckpt["student"], strict=False)
auc, acc, sep = eval_model(model, ds, DEVICE)
print(f"\n  AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, "results.txt"), "a") as f:
    f.write(f"{desc}\t{auc:.4f}\t{acc*100:.2f}\t{sep:.4f}\n")