"""
正式消融训练：在 CASIA-WebFace 上训练 6 个配置，用于论文实验。
6 配置构成完整消融链：baseline → +C1(channel/spatial/full) → +C2(distill) → +C3(aug)

用法（用户 GPU 环境）:
    python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256
    python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --configs 4 5

教师特征缓存：首次运行自动预计算 buffalo_s 教师特征并缓存，后续运行直接加载。
预计时间：6 配置 * 30 epoch * ~8min/epoch ≈ 24 小时（RTX 5060, bs=256）
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer
from qad_face.train.dataset import FaceClassDataset
from qad_face.data.surveillance_aug import SurveillanceAugment
from sklearn.metrics import roc_auc_score


ABLATION_CONFIGS = [
    ("x05_baseline",          {"w_feat": 0.0, "w_rel": 0.0, "use_aug": False, "distill_start": 0},  "baseline (ShuffleNetV2 x0.5)"),
    ("x05_sab_channel",       {"w_feat": 0.0, "w_rel": 0.0, "use_aug": False, "distill_start": 0},  "+C1 channel (ECA)"),
    ("x05_sab_spatial",       {"w_feat": 0.0, "w_rel": 0.0, "use_aug": False, "distill_start": 0},  "+C1 spatial"),
    ("x05_sab_full",          {"w_feat": 0.0, "w_rel": 0.0, "use_aug": False, "distill_start": 0},  "+C1 full SAB"),
    ("x05_sab_full",          {"w_feat": 0.1, "w_rel": 0.0, "use_aug": False, "distill_start": 20}, "+C1 +C2 distill"),
    ("x05_sab_full",          {"w_feat": 0.1, "w_rel": 0.0, "use_aug": True,  "distill_start": 20}, "+C1 +C2 +C3 aug"),
]


class AugFaceDataset(torch.utils.data.Dataset):
    """人脸分类数据集，可选 C3 监控退化增强。"""
    def __init__(self, root, use_aug=False, aug_prob=0.3, severity="light"):
        self.images = []
        self.labels = []
        self.use_aug = use_aug
        self.aug_prob = aug_prob
        self.augmentor = SurveillanceAugment(severity=severity) if use_aug else None
        import glob
        id_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        label = 0
        for name in id_dirs:
            imgs = sorted(glob.glob(os.path.join(root, name, "*.jpg")))
            if len(imgs) < 2:
                continue
            for p in imgs:
                self.images.append(p)
                self.labels.append(label)
            label += 1
        self.num_classes = label
        print(f"  dataset: {len(self.images)} imgs, {self.num_classes} ids, aug={use_aug}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imread_cn(self.images[idx])
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        if self.use_aug and np.random.random() < self.aug_prob:
            img = self.augmentor(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), self.labels[idx], idx


def precompute_teacher_cache(dataset, cache_path, batch_size=256):
    """预计算 buffalo_s 教师特征并缓存（批量 GPU 加速）。"""
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        print(f"  [Teacher] loaded cache: {feats.shape} from {cache_path}")
        return torch.from_numpy(feats).float()

    import onnxruntime as ort
    print(f"  [Teacher] precomputing for {len(dataset)} samples (batch={batch_size})...")
    onnx_path = "weights/models/buffalo_s/w600k_mbf.onnx"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    n = len(dataset)
    feats = np.zeros((n, 512), dtype=np.float32)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = np.zeros((end - i, 3, 112, 112), dtype=np.float32)
        for j in range(end - i):
            img = imread_cn(dataset.images[i + j])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
            batch[j] = np.transpose(img, (2, 0, 1))
        emb = sess.run([out_name], {inp_name: batch})[0]
        feats[i:end] = emb[:, :512]
        if (i + batch_size) % 5000 < batch_size:
            print(f"    {end}/{n}", flush=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, feats)
    print(f"  [Teacher] cached to {cache_path}: {feats.shape}")
    return torch.from_numpy(feats).float()


def eval_model(model, dataset, device, batch_size=256):
    """自验证：AUC + Acc + Sep（批量推理）。"""
    model.eval()
    feats = np.zeros((len(dataset), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            end = min(i + batch_size, len(dataset))
            batch = np.zeros((end - i, 3, 112, 112), dtype=np.float32)
            for j in range(end - i):
                img = imread_cn(dataset.images[i + j])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                batch[j] = np.transpose(img, (2, 0, 1))
            t = torch.from_numpy(batch).to(device)
            feats[i:end] = model.get_embedding(t).cpu().numpy()
    labels = np.array(dataset.labels)
    norms = np.linalg.norm(feats, axis=1) + 1e-8
    sims, same = [], []
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):
            sim = np.dot(feats[i], feats[j]) / (norms[i] * norms[j])
            sims.append(sim)
            same.append(1 if labels[i] == labels[j] else 0)
    sims, same = np.array(sims), np.array(same)
    auc = roc_auc_score(same, sims)
    best_acc = max(np.mean((sims >= t).astype(int) == same) for t in np.linspace(sims.min(), sims.max(), 300))
    sep = sims[same == 1].mean() - sims[same == 0].mean()
    return auc, best_acc, sep


def main():
    parser = argparse.ArgumentParser(description="消融训练")
    parser.add_argument("--data", required=True, help="对齐后数据集目录")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_dir", default="weights/ablation")
    parser.add_argument("--teacher_cache", default="weights/ablation/teacher_feats.npy")
    parser.add_argument("--configs", nargs="*", default=None, help="只跑指定配置（索引 0-5）")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复（路径）")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="梯度范数裁剪上限 (0=不裁剪)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="warmup epoch 数")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"], help="优化器")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, CUDA: {torch.cuda.is_available()}")

    configs = ABLATION_CONFIGS
    orig_indices = list(range(len(ABLATION_CONFIGS)))
    if args.configs:
        orig_indices = [int(x) for x in args.configs]
        configs = [ABLATION_CONFIGS[i] for i in orig_indices]

    teacher_feats = None
    results = []

    for idx, (cfg_name, params, desc) in enumerate(configs):
        orig_i = orig_indices[idx]
        tag = f"cfg{orig_i}_{desc.replace(' ', '').replace('+', '')}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(configs)}] {desc} (model={cfg_name}, cfg{orig_i})")
        print(f"  w_feat={params['w_feat']}, w_rel={params['w_rel']}, aug={params['use_aug']}, distill_start={params['distill_start']}")
        print(f"{'='*60}")

        ds = AugFaceDataset(args.data, use_aug=params["use_aug"])
        nw = args.num_workers if os.name != "nt" else min(args.num_workers, 2)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=nw, drop_last=True,
                        persistent_workers=(nw > 0),
                        pin_memory=(device != "cpu"))

        if params["w_feat"] > 0 or params["w_rel"] > 0:
            if teacher_feats is None:
                teacher_feats = precompute_teacher_cache(ds, args.teacher_cache)
        else:
            teacher_feats = None

        trainer = DistillTrainer(
            student_config=DEFAULT_CONFIGS[cfg_name],
            num_classes=ds.num_classes, device=device,
            w_arc=1.0, w_feat=params["w_feat"], w_rel=params["w_rel"],
            feat_mode="cosine",
            seed=args.seed,
        )

        def eval_wrapper(m, dev):
            auc_v, acc_v, sep_v = eval_model(m, ds, dev)
            return {"auc": auc_v, "acc": acc_v, "sep": sep_v}

        save_name = f"cfg{orig_i}_seed{args.seed}"
        t0 = time.time()
        resume_ckpt = args.resume if idx == 0 else None
        trainer.train(dl, teacher_feats, epochs=args.epochs, lr=args.lr,
                      save_dir=args.save_dir, save_name=save_name,
                      log_every=100, use_amp=(device != "cpu"),
                      grad_clip=args.grad_clip, resume_from=resume_ckpt,
                      warmup_epochs=args.warmup_epochs,
                      optimizer_type=args.optimizer,
                      distill_start_epoch=params["distill_start"],
                      seed=args.seed, eval_fn=eval_wrapper)

        model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(device)
        ckpt = torch.load(f"{args.save_dir}/{save_name}_final.pt", map_location=device)
        model.load_state_dict(ckpt["student"], strict=False)

        auc, acc, sep = eval_model(model, ds, device)
        elapsed = time.time() - t0
        print(f"\n  RESULT: AUC={auc:.4f} Acc={acc*100:.2f}% Sep={sep:.4f} ({elapsed:.0f}s)")
        results.append((desc, auc, acc, sep, elapsed))

    print(f"\n{'='*60}")
    print(f"  消融实验汇总")
    print(f"{'='*60}")
    print(f"  {'配置':<25} {'AUC':>8} {'Acc':>8} {'Sep':>8} {'时间':>8}")
    print(f"  {'-'*57}")
    for desc, auc, acc, sep, t in results:
        print(f"  {desc:<25} {auc:>8.4f} {acc*100:>7.2f}% {sep:>8.4f} {t:>7.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()