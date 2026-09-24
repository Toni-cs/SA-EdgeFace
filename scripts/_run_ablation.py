"""
完整 6 配置消融实验（face_dataset_52, 15 epoch, CPU）。
消融链：baseline → +C1(channel/spatial/full) → +C2(distill) → +C3(aug)
"""
import os, sys, time, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer
from qad_face.data.surveillance_aug import SurveillanceAugment
from sklearn.metrics import roc_auc_score

DST = "datasets/train_mini_if"
EPOCHS = 15
BATCH = 32

class FaceDS(Dataset):
    def __init__(self, root, use_aug=False):
        self.images, self.labels = [], []
        self.use_aug = use_aug
        self.aug = SurveillanceAugment() if use_aug else None
        for label, name in enumerate(sorted(os.listdir(root))):
            d = os.path.join(root, name)
            if not os.path.isdir(d): continue
            imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
            if len(imgs) < 2: continue
            for p in imgs:
                img = imread_cn(p)
                if img is None: continue
                self.images.append(p)
                self.labels.append(label)
        self.num_classes = label + 1
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = imread_cn(self.images[idx])
        if img is None: img = np.zeros((112,112,3), dtype=np.uint8)
        if self.use_aug and np.random.random() < 0.5:
            img = self.aug(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112)).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2,0,1))), self.labels[idx], idx

def precompute_teacher(ds):
    import onnxruntime as ort
    sess = ort.InferenceSession("weights/models/buffalo_s/w600k_mbf.onnx", providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    feats = np.zeros((len(ds), 512), dtype=np.float32)
    for i in range(len(ds)):
        img = imread_cn(ds.images[i])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112)).astype(np.float32) / 127.5 - 1.0
        feats[i] = sess.run([out], {inp: np.transpose(img,(2,0,1))[None]})[0].flatten()[:512]
    return torch.from_numpy(feats).float()

def eval_model(model, ds):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(len(ds)):
            img = imread_cn(ds.images[i])
            if img is None: feats.append(np.zeros(512)); continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112,112)).astype(np.float32) / 127.5 - 1.0
            t = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)
            feats.append(model.get_embedding(t).numpy().flatten())
    feats = np.array(feats)
    labels = np.array(ds.labels)
    norms = np.linalg.norm(feats, axis=1) + 1e-8
    sims, same = [], []
    for i in range(len(feats)):
        for j in range(i+1, min(i+50, len(feats))):
            sims.append(np.dot(feats[i], feats[j]) / (norms[i]*norms[j]))
            same.append(1 if labels[i]==labels[j] else 0)
    sims, same = np.array(sims), np.array(same)
    auc = roc_auc_score(same, sims)
    acc = max(np.mean((sims>=t).astype(int)==same) for t in np.linspace(sims.min(), sims.max(), 300))
    sep = sims[same==1].mean() - sims[same==0].mean()
    return auc, acc, sep

CONFIGS = [
    ("x05_baseline",    0.0, False, "baseline"),
    ("x05_sab_channel", 0.0, False, "+C1 channel"),
    ("x05_sab_spatial", 0.0, False, "+C1 spatial"),
    ("x05_sab_full",    0.0, False, "+C1 full SAB"),
    ("x05_sab_full",    0.1, False, "+C1+C2 distill"),
    ("x05_sab_full",    0.1, True,  "+C1+C2+C3 aug"),
]

t0 = time.time()
ds = FaceDS(DST)
print(f"dataset: {len(ds)} imgs, {ds.num_classes} ids", flush=True)
tf = precompute_teacher(ds)
print(f"teacher feats: {tf.shape}, nonzero: {(tf.norm(dim=1)>0).sum().item()}", flush=True)

results = []
for i, (cfg, wf, aug, desc) in enumerate(CONFIGS):
    ds_i = FaceDS(DST, use_aug=aug)
    dl = DataLoader(ds_i, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    trainer = DistillTrainer(
        DEFAULT_CONFIGS[cfg], ds_i.num_classes, device="cpu",
        w_arc=1.0, w_feat=wf, w_rel=0.0, feat_mode="cosine")
    print(f"\n[{i+1}/6] {desc} ({cfg}) w_feat={wf} aug={aug}", flush=True)
    trainer.train(dl, tf if wf > 0 else None, epochs=EPOCHS, lr=1e-3,
                  save_dir="weights/ablation_final", save_name=f"cfg{i}",
                  log_every=100, use_amp=False)
    model = build_edgeface(DEFAULT_CONFIGS[cfg])
    ckpt = torch.load(f"weights/ablation_final/cfg{i}_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=False)
    auc, acc, sep = eval_model(model, ds)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} Params={nparams:.2f}M", flush=True)
    results.append((desc, auc, acc, sep, nparams))

print(f"\n{'='*65}")
print(f"  消融实验结果 (face_dataset_52, {EPOCHS} epochs, CPU)")
print(f"{'='*65}")
print(f"  {'配置':<20} {'AUC':>8} {'Acc':>8} {'Sep':>8} {'Params':>8}")
print(f"  {'-'*52}")
for desc, auc, acc, sep, p in results:
    print(f"  {desc:<20} {auc:>8.4f} {acc*100:>7.2f}% {sep:>8.4f} {p:>7.2f}M")
print(f"{'='*65}")
print(f"[DONE] {time.time()-t0:.1f}s", flush=True)