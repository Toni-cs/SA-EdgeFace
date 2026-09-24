"""
真教师蒸馏验证：buffalo_s 教师 → sab_full 学生，在 face_dataset_52 上验证 C2 完整链路。
绕过 argparse（沙箱命令行受限），直接调用训练逻辑。
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
from qad_face.utils import imread_cn, imwrite_cn
from qad_face.models.recognizer import InsightFaceRecognizer
from qad_face.train.distill import DistillTrainer
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from sklearn.metrics import roc_auc_score

SRC = "datasets/train_mini_raw"
DST = "datasets/train_mini_if"
DEVICE = "cpu"
EPOCHS = 15


def align_with_insightface():
    print("[1] insightface 对齐", flush=True)
    rec = InsightFaceRecognizer(model_name="buffalo_s", root="weights")
    if os.path.isdir(DST) and len(os.listdir(DST)) > 40:
        print(f"  {DST} 已存在，跳过对齐", flush=True)
        return
    os.makedirs(DST, exist_ok=True)
    total, ok = 0, 0
    for ident in sorted(os.listdir(SRC)):
        id_dir = os.path.join(SRC, ident)
        if not os.path.isdir(id_dir):
            continue
        out_dir = os.path.join(DST, ident)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(id_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total += 1
            img = imread_cn(os.path.join(id_dir, f))
            if img is None:
                continue
            faces = rec.get_all_faces(img)
            if len(faces) == 0:
                continue
            face = max(faces, key=lambda x: x["det_score"])
            x1, y1, x2, y2 = face["bbox"].astype(int)
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (112, 112))
            imwrite_cn(os.path.join(out_dir, os.path.splitext(f)[0] + ".jpg"), crop,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            ok += 1
    print(f"  aligned {ok}/{total}", flush=True)


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
    """直接用 buffalo_s 的 w600k_mbf.onnx 识别模型提取特征，绕过检测器。"""
    import onnxruntime as ort
    print("[2] 预计算教师特征 (buffalo_s w600k_mbf.onnx)", flush=True)
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
            feats[idx] = emb[:512] if len(emb) >= 512 else np.pad(emb, (0, 512 - len(emb)))
            idx += 1
    print(f"  teacher feats: {feats.shape}, nonzero: {(np.linalg.norm(feats, axis=1) > 0).sum()}", flush=True)
    return torch.from_numpy(feats).float()


def train_and_eval(teacher_feats, ds, w_feat, w_rel, save_name, tag):
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    trainer = DistillTrainer(
        student_config=DEFAULT_CONFIGS["sab_full"],
        num_classes=ds.num_classes, device=DEVICE,
        w_arc=1.0, w_feat=w_feat, w_rel=w_rel,
    )
    print(f"[3] 训练 {tag} ({EPOCHS} epochs)", flush=True)
    trainer.train(dl, teacher_feats, epochs=EPOCHS, lr=1e-3,
                  save_dir="weights/distill_real", save_name=save_name,
                  log_every=10, use_amp=False)

    model = build_edgeface(DEFAULT_CONFIGS["sab_full"])
    ckpt = torch.load(f"weights/distill_real/{save_name}_final.pt", map_location="cpu")
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
    print(f"\n  [{tag}] AUC={auc:.4f} Acc={best_acc*100:.2f}% Sep={sep:.4f}", flush=True)
    return auc, best_acc, sep


def main():
    t0 = time.time()
    align_with_insightface()
    ds = InMemoryFaceDataset(DST)
    print(f"  dataset: {len(ds)} imgs, {ds.num_classes} ids, {time.time()-t0:.1f}s", flush=True)

    teacher_feats = precompute_teacher(ds)

    print("\n=== A: 无蒸馏 (纯 ArcFace) ===", flush=True)
    auc_a, acc_a, sep_a = train_and_eval(teacher_feats, ds, 0.0, 0.0, "no_distill", "no-distill")

    print("\n=== B: 真蒸馏 (buffalo_s 教师) ===", flush=True)
    auc_b, acc_b, sep_b = train_and_eval(teacher_feats, ds, 0.5, 0.1, "with_distill", "with-distill")

    print(f"\n{'='*55}")
    print(f"  C2 蒸馏验证对比")
    print(f"{'='*55}")
    print(f"  无蒸馏:   AUC={auc_a:.4f} Acc={acc_a*100:.2f}% Sep={sep_a:.4f}")
    print(f"  有蒸馏:   AUC={auc_b:.4f} Acc={acc_b*100:.2f}% Sep={sep_b:.4f}")
    print(f"  蒸馏增益: AUC +{auc_b-auc_a:.4f}  Acc +{(acc_b-acc_a)*100:.2f}%")
    print(f"{'='*55}")
    print(f"[DONE] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()