import time, os
from PIL import Image
import numpy as np

CASIA = "datasets/raw/casia-webface"
paths = []
for d in sorted(os.listdir(CASIA))[:10]:
    for f in sorted(os.listdir(os.path.join(CASIA, d)))[:5]:
        paths.append(os.path.join(CASIA, d, f))

t0 = time.time()
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    arr = np.array(img)
print(f"{len(paths)} imgs in {time.time()-t0:.2f}s = {(time.time()-t0)/len(paths)*1000:.1f}ms/img", flush=True)

# 测 imread_cn
import sys
sys.path.insert(0, ".")
from qad_face.utils import imread_cn
t0 = time.time()
for p in paths:
    img = imread_cn(p)
print(f"imread_cn: {len(paths)} imgs in {time.time()-t0:.2f}s = {(time.time()-t0)/len(paths)*1000:.1f}ms/img", flush=True)

# 测 cv2.imdecode + np.fromfile
import cv2
t0 = time.time()
for p in paths:
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
print(f"cv2.imdecode: {len(paths)} imgs in {time.time()-t0:.2f}s = {(time.time()-t0)/len(paths)*1000:.1f}ms/img", flush=True)