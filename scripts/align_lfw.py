"""
LFW 人脸对齐：用 insightface buffalo_s 检测 + ArcFace 5点模板对齐到 112×112。
训练数据 CASIA 已对齐，评测数据 LFW 未对齐（250×250）导致分布不匹配。
对齐后无需重训，直接重新评测。
用法: python scripts/align_lfw.py
"""
import os, sys, glob, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.utils import imread_cn, imwrite_cn
from insightface.app import FaceAnalysis
from insightface.utils import face_align

LFW = "datasets/lfw"
OUT = "datasets/lfw_aligned"

app = FaceAnalysis(name="buffalo_s", root="weights",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

id_dirs = sorted([d for d in os.listdir(LFW) if os.path.isdir(os.path.join(LFW, d))])
print(f"LFW: {len(id_dirs)} id dirs")
t0 = time.time()
done = fail = skip = 0
for d in id_dirs:
    out_dir = os.path.join(OUT, d)
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(LFW, d, "*.jpg"))):
        out_path = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(out_path):
            skip += 1
            continue
        img = imread_cn(f)
        if img is None:
            fail += 1
            continue
        faces = app.get(img)
        if len(faces) == 0:
            imwrite_cn(out_path, cv2.resize(img, (112, 112)))
            fail += 1
            continue
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        aligned = face_align.norm_crop(img, face.kps)
        imwrite_cn(out_path, aligned)
        done += 1
        if done % 1000 == 0:
            print(f"  aligned={done} nodetect={fail} skipped={skip} ({time.time()-t0:.0f}s)", flush=True)

print(f"[done] aligned={done} nodetect_fallback={fail} skipped_existing={skip} ({time.time()-t0:.0f}s)")
print(f"output: {OUT}")