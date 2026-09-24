"""
CASIA 1000 身份子集 GPU 训练：6 配置消融，分批跑。
用法: python scripts/_casia_subset.py 0  # 跑第 0 个配置
      python scripts/_casia_subset.py all  # 跑全部 6 个
"""
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
N_IDS = 1000
SAVE_DIR = "weights/ablation_casia1k"
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
            d_dir = os.path.join(root, d)
            for f in os.listdir(d_dir):
                if f.endswith(".jpg"):
                    self.paths.append(os.path.join(d_dir, f))
                    self.labels.append(label)
        self.num_classes = n_ids

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img, dtype=np.uint8)
        if self.use_aug and np.random.random() < 0.5:
            img = self.aug(img)
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), self.labels[idx], idx


def precompute_teacher(ds):
    if os.path.exists(TEACHER_CACHE):
        print(f"[Teacher] cache: {TEACHER_CACHE}")
        return torch.from_numpy(np.load(TEACHER_CACHE)).float()
    import onnxruntime as ort
    print(f"[Teacher] precomputing for {len(ds)} images...", flush=True)
    sess = ort.InferenceSession("weights/models/buffalo_s/w600k_mbf.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    feats = np.zeros((len(ds), 512), dtype=np.float32)
    bs = 256
    t0 = time.time()
    for i in range(0, len(ds), bs):
        batch = np.zeros((min(bs, len(ds)-i), 3, 112, 112), dtype=np.float32)
        for j in range(len(batch)):
            img = Image.open(ds.paths[i+j]).convert("RGB")
            batch[j] = np.transpose(np.array(img, dtype=np.float32)/127.5-1.0, (2,0,1))
        feats[i:i+len(batch)] = sess.run([out], {inp: batch})[0][:, :512]
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TEACHER_CACHE, feats)
    print(f"  done: {feats.shape} ({time.time()-t0:.0f}s)", flush=True)
    return torch.from_numpy(feats).float()


def eval_model(model, ds, device):
    model.eval()
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
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


def train_one(idx):
    cfg_name, w_feat, use_aug, desc = CONFIGS[idx]
    print(f"\n[{idx+1}/6] {desc} ({cfg_name}) w_feat={w_feat} aug={use_aug}", flush=True)
    ds = CasiaSubset(CASIA, N_IDS, use_aug=use_aug)
    print(f"  {len(ds)} imgs, {ds.num_classes} ids", flush=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    tf = precompute_teacher(ds) if w_feat > 0 else None
    trainer = DistillTrainer(DEFAULT_CONFIGS[cfg_name], ds.num_classes, device=DEVICE,
                             w_arc=1.0, w_feat=w_feat, w_rel=0.0, feat_mode="cosine")
    t0 = time.time()
    trainer.train(dl, tf, epochs=EPOCHS, lr=1e-3,
                  save_dir=SAVE_DIR, save_name=f"cfg{idx}",
                  log_every=100, use_amp=True)
    model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(DEVICE)
    ckpt = torch.load(f"{SAVE_DIR}/cfg{idx}_final.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["student"], strict=False)
    auc, acc, sep = eval_model(model, ds, DEVICE)
    print(f"  AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(SAVE_DIR, "results.txt"), "a") as f:
        f.write(f"{desc}\t{auc:.4f}\t{acc*100:.2f}\t{sep:.4f}\n")
    return auc, acc, sep


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        results = []
        for i in range(6):
            results.append((CONFIGS[i][3], *train_one(i)))
        print(f"\n{'='*60}")
        print(f"  CASIA {N_IDS}id 消融结果 ({EPOCHS} epochs, GPU)")
        print(f"{'='*60}")
        for desc, auc, acc, sep in results:
            print(f"  {desc:<20} AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f}")
        print(f"{'='*60}")
    else:
        train_one(int(arg))