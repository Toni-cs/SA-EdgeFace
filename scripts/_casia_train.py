"""
CASIA 全量 GPU 训练：支持分批跑单个配置，checkpoint 自动保存/恢复。
用法: python scripts/_casia_train.py 0  # 跑第 0 个配置
      python scripts/_casia_train.py 1  # 跑第 1 个配置
      python scripts/_casia_train.py teacher  # 预计算教师特征
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
EPOCHS = 10
BATCH = 256
SAVE_DIR = "weights/ablation_casia"
TEACHER_CACHE = os.path.join(SAVE_DIR, "teacher_feats.npy")

CONFIGS = [
    ("x05_baseline",    0.0, False, "baseline"),
    ("x05_sab_channel", 0.0, False, "+C1 channel"),
    ("x05_sab_spatial", 0.0, False, "+C1 spatial"),
    ("x05_sab_full",    0.0, False, "+C1 full SAB"),
    ("x05_sab_full",    0.1, False, "+C1+C2 distill"),
    ("x05_sab_full",    0.1, True,  "+C1+C2+C3 aug"),
]


class CasiaDataset(Dataset):
    def __init__(self, root, use_aug=False, aug_prob=0.5):
        self.paths, self.labels = [], []
        self.use_aug = use_aug
        self.aug = SurveillanceAugment() if use_aug else None
        self.aug_prob = aug_prob
        dirs = sorted(os.listdir(root))
        for label, d in enumerate(dirs):
            d_dir = os.path.join(root, d)
            if not os.path.isdir(d_dir):
                continue
            for f in os.listdir(d_dir):
                if f.endswith(".jpg"):
                    self.paths.append(os.path.join(d_dir, f))
                    self.labels.append(label)
        self.num_classes = len(dirs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img, dtype=np.uint8)
        if self.use_aug and np.random.random() < self.aug_prob:
            import cv2
            img = self.aug(img)
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), self.labels[idx], idx


def precompute_teacher():
    if os.path.exists(TEACHER_CACHE):
        print(f"[Teacher] cache exists: {TEACHER_CACHE}")
        return torch.from_numpy(np.load(TEACHER_CACHE)).float()

    import onnxruntime as ort
    print("[Teacher] precomputing buffalo_s features for all CASIA images...")
    sess = ort.InferenceSession(
        "weights/models/buffalo_s/w600k_mbf.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    dirs = sorted(os.listdir(CASIA))
    all_paths = []
    for d in dirs:
        d_dir = os.path.join(CASIA, d)
        if not os.path.isdir(d_dir):
            continue
        for f in os.listdir(d_dir):
            if f.endswith(".jpg"):
                all_paths.append(os.path.join(d_dir, f))

    feats = np.zeros((len(all_paths), 512), dtype=np.float32)
    bs = 256
    t0 = time.time()
    for i in range(0, len(all_paths), bs):
        batch_paths = all_paths[i:i+bs]
        batch = np.zeros((len(batch_paths), 3, 112, 112), dtype=np.float32)
        for j, p in enumerate(batch_paths):
            img = Image.open(p).convert("RGB")
            img = np.array(img, dtype=np.float32) / 127.5 - 1.0
            batch[j] = np.transpose(img, (2, 0, 1))
        emb = sess.run([out_name], {inp_name: batch})[0]
        feats[i:i+len(batch_paths)] = emb[:, :512]
        if (i // bs + 1) % 100 == 0:
            elapsed = time.time() - t0
            pct = (i + bs) / len(all_paths) * 100
            eta = elapsed / (i + bs) * (len(all_paths) - i - bs)
            print(f"  {i+bs}/{len(all_paths)} ({pct:.1f}%) {elapsed:.0f}s eta {eta:.0f}s", flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TEACHER_CACHE, feats)
    print(f"[Teacher] saved {feats.shape} to {TEACHER_CACHE} ({time.time()-t0:.0f}s)")
    return torch.from_numpy(feats).float()


def eval_model(model, ds, device):
    model.eval()
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in dl:
            imgs = imgs.to(device)
            f = model.get_embedding(imgs).cpu().numpy()
            feats.append(f)
            labels.append(lbls.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    norms = np.linalg.norm(feats, axis=1) + 1e-8
    sims, same = [], []
    n = len(feats)
    for i in range(0, n, 7):
        for j in range(i+1, min(i+100, n)):
            sims.append(np.dot(feats[i], feats[j]) / (norms[i]*norms[j]))
            same.append(1 if labels[i]==labels[j] else 0)
    sims, same = np.array(sims), np.array(same)
    if len(set(same)) < 2:
        return 0.0, 0.0, 0.0
    auc = roc_auc_score(same, sims)
    acc = max(np.mean((sims>=t).astype(int)==same) for t in np.linspace(sims.min(), sims.max(), 200))
    sep = sims[same==1].mean() - sims[same==0].mean()
    return auc, acc, sep


def train_one(idx):
    cfg_name, w_feat, use_aug, desc = CONFIGS[idx]
    save_name = f"cfg{idx}"
    print(f"\n{'='*60}")
    print(f"[{idx+1}/6] {desc} ({cfg_name}) w_feat={w_feat} aug={use_aug}")
    print(f"{'='*60}", flush=True)

    ds = CasiaDataset(CASIA, use_aug=use_aug)
    print(f"dataset: {len(ds)} imgs, {ds.num_classes} ids", flush=True)

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    tf = None
    if w_feat > 0:
        tf = precompute_teacher()

    trainer = DistillTrainer(
        DEFAULT_CONFIGS[cfg_name], ds.num_classes, device=DEVICE,
        w_arc=1.0, w_feat=w_feat, w_rel=0.0, feat_mode="cosine")

    t0 = time.time()
    trainer.train(dl, tf, epochs=EPOCHS, lr=1e-3,
                  save_dir=SAVE_DIR, save_name=save_name,
                  log_every=200, use_amp=True)

    model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(DEVICE)
    ckpt = torch.load(f"{SAVE_DIR}/{save_name}_final.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["student"], strict=False)
    auc, acc, sep = eval_model(model, ds, DEVICE)
    print(f"\n  RESULT: AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(SAVE_DIR, "results.txt"), "a") as f:
        f.write(f"{desc}\t{auc:.4f}\t{acc*100:.2f}\t{sep:.4f}\t{time.time()-t0:.0f}\n")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    if len(sys.argv) < 2:
        print("Usage: python _casia_train.py <index 0-5 | teacher | all>")
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "teacher":
        precompute_teacher()
    elif arg == "all":
        for i in range(6):
            train_one(i)
    else:
        train_one(int(arg))