"""快速检查 CASIA 图片尺寸（20 身份）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.utils import imread_cn
import numpy as np

casia = "datasets/raw/casia-webface"
dirs = sorted(os.listdir(casia))[:20]
sizes = []
for d in dirs:
    for f in sorted(os.listdir(os.path.join(casia, d)))[:2]:
        img = imread_cn(os.path.join(casia, d, f))
        if img is not None:
            sizes.append(img.shape[:2])

sizes = np.array(sizes)
print(f"sample: {len(sizes)} imgs")
print(f"H: min={sizes[:,0].min()}, max={sizes[:,0].max()}, mean={sizes[:,0].mean():.0f}")
print(f"W: min={sizes[:,1].min()}, max={sizes[:,1].max()}, mean={sizes[:,1].mean():.0f}")
print(f"square: {(sizes[:,0]==sizes[:,1]).sum()}/{len(sizes)}")
