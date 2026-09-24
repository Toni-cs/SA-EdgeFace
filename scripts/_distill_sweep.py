"""
蒸馏超参搜索：在 face_dataset_52 上测试多组 (w_feat, w_rel, feat_mode) 配置，
找到使蒸馏增益最大的组合。修复 C2 验证的关键脚本。
"""

import os
import sys
import time
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.train.distill import DistillTrainer
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from sklearn.metrics import roc_auc_score

DST = "datasets/train_mini_if"
DEVICE = "cpu"
EPOCHS = 15


class InMemoryFaceDataset(Dataset):
    def __init__(self, root):
        self.images, self.labels = [], []
        id_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        label = 0
        for name in id_dirs:
            imgs = sorted(glob.glob(os.path.join(root, name, "*.jpg")))
            if len(imgs) < 2:
                continue
            for p in imgs:
                img = imread_cn(p)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                self.images.append(np.transpose(img, (2, 0, 1)))
                self.labels.append(label)
            label += 1
        self.num_classes = label
        self.paths = [None] * len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), self.labels[idx], idx


def precompute_teacher(ds):
    import onnxruntime as ort
    print("[Teacher] 预计算 buffalo_s 特征", flush=True)
    onnx_path = "weights/models/buffalo_s/w600k_mbf.onnx"
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    feats = np.zeros((len(ds), 512), dtype=np.float32)
    id_dirs = sorted([d for d in os.listdir(DST) if os.path.isdir(os.path.join(DST, d))])
    idx = 0
    for name in id_dirs:
        imgs = sorted(glob.glob(os.path.join(DST, name, "*.jpg")))
        if len(imgs) < 2:
            continue
        for p in imgs:
            img = imread_cn(p)
            if img is None:
                idx += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
            img = np.transpose(img, (2, 0, 1))[None]
            emb = sess.run([out_name], {inp_name: img})[0].flatten()
            feats[idx] = emb[:512]
            idx += 1
    print(f"  feats: {feats.shape}, nonzero: {(np.linalg.norm(feats, axis=1) > 0).sum()}", flush=True)
    return torch.from_numpy(feats).float()


def train_and_eval(teacher_feats, ds, w_feat, w_rel, feat_mode, save_name, tag):
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    trainer = DistillTrainer(
        student_config=DEFAULT_CONFIGS["sab_full"],
        num_classes=ds.num_classes, device=DEVICE,
        w_arc=1.0, w_feat=w_feat, w_rel=w_rel, feat_mode=feat_mode,
    )
    print(f"\n[Train] {tag} w_feat={w_feat} w_rel={w_rel} mode={feat_mode}", flush=True)
    trainer.train(dl, teacher_feats, epochs=EPOCHS, lr=1e-3,
                  save_dir="weights/distill_sweep", save_name=save_name,
                  log_every=100, use_amp=False)

    model = build_edgeface(DEFAULT_CONFIGS["sab_full"])
    ckpt = torch.load(f"weights/distill_sweep/{save_name}_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    with torch.no_grad():
        feats = []
        for i in range(len(ds)):
            t = torch.from_numpy(ds.images[i]).unsqueeze(0)
            feats.append(model.get_embedding(t).numpy().flatten())
    feats = np.array(feats)
    labels = np.array(ds.labels)
    norms = np.linalg.norm(feats, axis=1)
    sims, same = [], []
    n = len(feats)
    for i in range(0, n, 2):
        for j in range(i + 1, min(i + 40, n)):
            sim = np.dot(feats[i], feats[j]) / (norms[i] * norms[j] + 1e-8)
            sims.append(sim)
            same.append(1 if labels[i] == labels[j] else 0)
    sims, same = np.array(sims), np.array(same)
    auc = roc_auc_score(same, sims)
    best_acc = max(np.mean((sims >= t).astype(int) == same) for t in np.linspace(sims.min(), sims.max(), 300))
    sep = sims[same == 1].mean() - sims[same == 0].mean()
    print(f"  [{tag}] AUC={auc:.4f} Acc={best_acc*100:.2f}% Sep={sep:.4f}", flush=True)
    return auc, best_acc, sep


def main():
    t0 = time.time()
    ds = InMemoryFaceDataset(DST)
    print(f"dataset: {len(ds)} imgs, {ds.num_classes} ids", flush=True)
    teacher_feats = precompute_teacher(ds)

    configs = [
        (0.0, 0.0, "cosine", "baseline", "no-distill"),
        (0.0, 0.1, "cosine", "rel_only", "rel-only"),
        (0.1, 0.1, "cosine", "feat01_rel01", "feat0.1+rel0.1"),
        (0.1, 0.0, "cosine", "feat01_only", "feat0.1-only"),
        (0.3, 0.05, "mse", "mse03_rel005", "mse0.3+rel0.05"),
        (0.5, 0.1, "mse", "mse05_rel01", "mse0.5+rel0.1"),
    ]

    results = []
    for w_feat, w_rel, mode, save_name, tag in configs:
        auc, acc, sep = train_and_eval(teacher_feats, ds, w_feat, w_rel, mode, save_name, tag)
        results.append((tag, auc, acc, sep))

    print(f"\n{'='*60}")
    print(f"  C2 蒸馏超参搜索结果")
    print(f"{'='*60}")
    print(f"  {'配置':<20} {'AUC':>8} {'Acc':>8} {'Sep':>8}")
    print(f"  {'-'*44}")
    base_auc = results[0][1]
    for tag, auc, acc, sep in results:
        delta = f"(+{auc-base_auc:.4f})" if auc > base_auc else f"({auc-base_auc:.4f})"
        print(f"  {tag:<20} {auc:>8.4f} {acc*100:>7.2f}% {sep:>8.4f}  {delta}")
    print(f"{'='*60}")
    print(f"[DONE] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()