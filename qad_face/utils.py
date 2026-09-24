"""
中文路径安全的 OpenCV 读写工具。

Windows 下 cv2.imread/imwrite/CascadeClassifier 对非 ASCII 路径会失败，
统一用 numpy fromfile/tofile + imencode/imdecode 绕过。
"""

import os
import shutil
import tempfile

import cv2
import numpy as np


def imread_cn(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_cn(path, img, params=None):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


def load_cascade_ascii(xml_path, cache_name=None):
    """把 XML 复制到 ASCII 临时目录后加载 CascadeClassifier。"""
    cache_name = cache_name or os.path.basename(xml_path)
    tmp = os.path.join(tempfile.gettempdir(), cache_name)
    if not os.path.exists(tmp):
        shutil.copy2(xml_path, tmp)
    cascade = cv2.CascadeClassifier(tmp)
    return cascade