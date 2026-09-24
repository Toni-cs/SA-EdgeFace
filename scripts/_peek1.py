"""极简检查：1 张 CASIA 图片尺寸。"""
from PIL import Image
import os
p = "datasets/raw/casia-webface/000000/00000001.jpg"
img = Image.open(p)
print(f"size: {img.size}, mode: {img.mode}")