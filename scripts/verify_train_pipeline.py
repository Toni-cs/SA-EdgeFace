"""
训练 pipeline 冒烟验证：用合成数据验证 DistillTrainer 训练循环无 bug。
不依赖 insightface 教师下载（用随机教师特征），仅验证训练代码正确性。

用法: python scripts/verify_train_pipeline.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.train.distill import DistillTrainer
from qad_face.models.edgeface import DEFAULT_CONFIGS


class IndexedTensorDataset(TensorDataset):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    num_samples, num_classes, emb_dim = 64, 8, 512
    imgs = torch.randn(num_samples, 3, 112, 112)
    labels = torch.randint(0, num_classes, (num_samples,))
    ds = IndexedTensorDataset(imgs, labels)
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

    teacher_feats = torch.randn(num_samples, emb_dim)

    trainer = DistillTrainer(
        student_config=DEFAULT_CONFIGS["sab_full"],
        num_classes=num_classes, device=device,
        w_arc=1.0, w_feat=0.5, w_rel=0.1,
    )

    save_dir = os.path.join("results", "smoke_train")
    trainer.train(dl, teacher_feats, epochs=2, lr=1e-3,
                  save_dir=save_dir, save_name="smoke", log_every=2, use_amp=False)

    ckpt_path = os.path.join(save_dir, "smoke_final.pt")
    assert os.path.exists(ckpt_path), "checkpoint not saved"
    print(f"[OK] 训练 pipeline 冒烟通过，checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()