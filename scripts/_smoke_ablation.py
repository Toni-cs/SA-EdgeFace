"""冒烟测试 train_ablation.py：用 face_dataset_52 跑前 3 个配置各 2 epoch。"""
import os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, glob
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.train.distill import DistillTrainer
from qad_face.data.surveillance_aug import SurveillanceAugment

DST = "datasets/train_mini_if"
DEVICE = "cpu"
EPOCHS = 2

class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, use_aug=False):
        self.images, self.labels = [], []
        self.use_aug = use_aug
        self.augmentor = SurveillanceAugment() if use_aug else None
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
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = imread_cn(self.images[idx])
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        if self.use_aug and np.random.random() < 0.5:
            img = self.augmentor(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), self.labels[idx], idx

def precompute_teacher(ds):
    import onnxruntime as ort
    sess = ort.InferenceSession("weights/models/buffalo_s/w600k_mbf.onnx", providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    feats = np.zeros((len(ds), 512), dtype=np.float32)
    for i in range(len(ds)):
        img = imread_cn(ds.images[i])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
        feats[i] = sess.run([out], {inp: np.transpose(img, (2, 0, 1))[None]})[0].flatten()[:512]
    return torch.from_numpy(feats).float()

configs = [
    ("x05_baseline",    0.0, 0.0, False, "baseline"),
    ("x05_sab_full",    0.0, 0.0, False, "+SAB"),
    ("x05_sab_full",    0.1, 0.0, False, "+SAB+distill"),
    ("x05_sab_full",    0.1, 0.0, True,  "+SAB+distill+aug"),
]

t0 = time.time()
for cfg, wf, wr, aug, tag in configs:
    ds = MiniDataset(DST, use_aug=aug)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    tf = precompute_teacher(ds) if wf > 0 else None
    trainer = DistillTrainer(
        student_config=DEFAULT_CONFIGS[cfg], num_classes=ds.num_classes,
        device=DEVICE, w_arc=1.0, w_feat=wf, w_rel=wr, feat_mode="cosine")
    print(f"\n=== {tag} ({cfg}) w_feat={wf} aug={aug} ===", flush=True)
    trainer.train(dl, tf, epochs=EPOCHS, lr=1e-3,
                  save_dir="weights/ablation_smoke", save_name=tag.replace("+","_"),
                  log_every=50, use_amp=False)
    n_params = sum(p.numel() for p in trainer.student.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

print(f"\n[DONE] {time.time()-t0:.1f}s")