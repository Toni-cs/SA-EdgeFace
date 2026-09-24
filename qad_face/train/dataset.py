"""
训练数据集：人脸分类数据集（identity folders 结构）。

目录组织:
    root/
    ├── identity_0/
    │   ├── 0001.jpg
    │   └── ...
    ├── identity_1/
    └── ...

支持 C3 SurveillanceAugment，输出 (tensor, label)。
对应论文：Section 4.1（训练数据）。
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from ..data.surveillance_aug import SurveillanceAugment
from ..utils import imread_cn


def collect_identity_samples(root, min_samples=2, max_identities=None):
    """扫描 root，返回 (image_paths, labels, identity_names)。label 为过滤后连续重编号。"""
    paths, labels, names = [], [], []
    id_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    label = 0
    for name in id_dirs:
        if max_identities is not None and label >= max_identities:
            break
        imgs = sorted(glob.glob(os.path.join(root, name, "*.jpg")) +
                      glob.glob(os.path.join(root, name, "*.png")))
        if len(imgs) < min_samples:
            continue
        for p in imgs:
            paths.append(p)
            labels.append(label)
            names.append(name)
        label += 1
    return paths, labels, names


class FaceClassDataset(Dataset):
    """人脸分类数据集，支持监控退化增强。"""

    def __init__(self, root, size=112, augment=None, min_samples=2,
                 max_identities=None, normalize=True):
        self.size = size
        self.augment = augment
        self.normalize = normalize
        self.paths, self.labels, self.names = collect_identity_samples(
            root, min_samples=min_samples, max_identities=max_identities
        )
        self.num_classes = len(set(self.labels))
        if len(self.paths) == 0:
            raise RuntimeError(f"No valid samples found under {root}")

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        img = imread_cn(path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size))
        return img

    def __getitem__(self, idx):
        img = self._load(self.paths[idx])
        if self.augment is not None:
            img = self.augment(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        if self.normalize:
            img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), self.labels[idx]


def build_train_dataset(root, severity="medium", use_aug=True, size=112, **kwargs):
    """构建带监控退化增强的训练数据集。"""
    aug = SurveillanceAugment(severity=severity) if use_aug else None
    return FaceClassDataset(root, size=size, augment=aug, **kwargs)