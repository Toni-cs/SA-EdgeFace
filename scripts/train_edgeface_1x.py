"""
训练 EdgeFace 1.0x baseline (ShuffleNetV2 1.0x, 无 SAB, 无蒸馏, 无增强)。
用于论文对比表：同架构 1.0x 宽度 vs 我们的 0.5x + SAB + 蒸馏。

用法（用户本地 GPU）:
    python scripts/train_edgeface_1x.py --data datasets/raw/casia-webface --gpu 0

预计时间：~4 小时 (8min/epoch × 30 epoch, RTX 5060, bs=256)
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
from sklearn.metrics import roc_auc_score


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.images = []
        self.labels = []
        import glob
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
                self.images.append(p)
                self.labels.append(label)
            label += 1
        self.num_classes = label
        print(f"  dataset: {len(self.images)} imgs, {self.num_classes} ids")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imread_cn(self.images[idx])
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), self.labels[idx], idx


def main():
    parser = argparse.ArgumentParser(description="训练 EdgeFace 1.0x baseline")
    parser.add_argument("--data", required=True, help="CASIA-WebFace 对齐后目录")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_dir", default="weights/baselines")
    parser.add_argument("--save_name", default="edgeface_1x")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, CUDA: {torch.cuda.is_available()}")

    cfg = DEFAULT_CONFIGS["baseline"]
    print(f"Config: baseline (width_mult=1.0, use_sab=False)")

    ds = FaceDataset(args.data)
    nw = args.num_workers if os.name != "nt" else min(args.num_workers, 2)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=nw, drop_last=True,
                    persistent_workers=(nw > 0),
                    pin_memory=(device != "cpu"))

    trainer = DistillTrainer(
        student_config=cfg,
        num_classes=ds.num_classes, device=device,
        w_arc=1.0, w_feat=0.0, w_rel=0.0,
        feat_mode="cosine",
    )

    print(f"\n开始训练: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")
    print(f"预计 ~{args.epochs * 8} 分钟 (8min/epoch)")

    t0 = time.time()
    trainer.train(dl, teacher_features=None, epochs=args.epochs, lr=args.lr,
                  save_dir=args.save_dir, save_name=args.save_name,
                  log_every=100, use_amp=(device != "cpu"),
                  grad_clip=args.grad_clip, resume_from=None,
                  warmup_epochs=args.warmup_epochs,
                  optimizer_type="sgd",
                  distill_start_epoch=0)
    elapsed = time.time() - t0
    print(f"\n训练完成: {elapsed/60:.1f} 分钟")
    print(f"模型保存到: {args.save_dir}/{args.save_name}_final.pt")

    print(f"\n下一步 - LFW 评测:")
    print(f"  python scripts/eval_model.py --checkpoint {args.save_dir}/{args.save_name}_final.pt --config baseline --lfw --lfw_dir datasets/lfw_aligned --gpu {args.gpu}")


if __name__ == "__main__":
    main()