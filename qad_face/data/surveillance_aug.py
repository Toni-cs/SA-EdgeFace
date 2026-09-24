"""
C3 辅创新：监控场景退化数据增强 pipeline。

模拟监控图像的退化因素，提升模型在低质量监控人脸上的泛化：
- 低分辨率降采样（监控小人脸）
- 高斯模糊（镜头/运动模糊）
- 随机遮挡（监控遮挡）
- JPEG 压缩伪影（传输压缩）
- 亮度/对比度抖动（监控光照变化）

基于 numpy + cv2，无额外依赖。可单独用于训练增强，也可用于评测时的退化模拟。
对应论文：Section 3.4（C3）。
"""

import numpy as np
import cv2


def random_downscale(img, min_size=40, prob=0.5):
    """随机降采样再上采样回原尺寸，模拟低分辨率监控人脸。"""
    if np.random.rand() > prob:
        return img
    h, w = img.shape[:2]
    scale = np.random.uniform(min_size / max(h, w), 0.8)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def random_blur(img, max_ksize=5, prob=0.5):
    """随机高斯模糊，模拟镜头/运动模糊。"""
    if np.random.rand() > prob:
        return img
    k = np.random.choice([3, 5, 7])
    k = min(k, max_ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def random_occlusion(img, max_ratio=0.15, prob=0.4):
    """随机小块遮挡，模拟监控遮挡。"""
    if np.random.rand() > prob:
        return img
    h, w = img.shape[:2]
    area = h * w * np.random.uniform(0.02, max_ratio)
    aspect = np.random.uniform(0.5, 2.0)
    oh = int(np.sqrt(area / aspect))
    ow = int(aspect * oh)
    oh, ow = min(oh, h - 1), min(ow, w - 1)
    if oh < 1 or ow < 1:
        return img
    y = np.random.randint(0, h - oh)
    x = np.random.randint(0, w - ow)
    img = img.copy()
    img[y:y + oh, x:x + ow] = np.random.randint(0, 256, size=(oh, ow, img.shape[2]), dtype=img.dtype)
    return img


def random_jpeg(img, quality_range=(20, 60), prob=0.4):
    """随机 JPEG 压缩，模拟传输压缩伪影。"""
    if np.random.rand() > prob:
        return img
    q = int(np.random.uniform(*quality_range))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def random_lighting(img, brightness=0.2, contrast=0.2, prob=0.5):
    """随机亮度/对比度抖动，模拟监控光照变化。"""
    if np.random.rand() > prob:
        return img
    img = img.astype(np.float32)
    b = np.random.uniform(-brightness, brightness) * 255
    c = 1.0 + np.random.uniform(-contrast, contrast)
    img = (img - 127.5) * c + 127.5 + b
    return np.clip(img, 0, 255).astype(np.uint8)


class SurveillanceAugment:
    """
    监控退化增强组合。可配置每种退化的开关与概率，支持消融。

    用法:
        aug = SurveillanceAugment(severity='medium')
        out = aug(img)  # img: BGR uint8 HWC
    """

    SEVERITY_PROB = {
        "none": {"downscale": 0, "blur": 0, "occlusion": 0, "jpeg": 0, "lighting": 0},
        "light": {"downscale": 0.3, "blur": 0.3, "occlusion": 0.2, "jpeg": 0.2, "lighting": 0.3},
        "medium": {"downscale": 0.5, "blur": 0.5, "occlusion": 0.4, "jpeg": 0.4, "lighting": 0.5},
        "heavy": {"downscale": 0.7, "blur": 0.7, "occlusion": 0.6, "jpeg": 0.6, "lighting": 0.7},
    }

    def __init__(self, severity="medium", use_downscale=True, use_blur=True,
                 use_occlusion=True, use_jpeg=True, use_lighting=True):
        probs = self.SEVERITY_PROB[severity]
        self.use_downscale = use_downscale
        self.use_blur = use_blur
        self.use_occlusion = use_occlusion
        self.use_jpeg = use_jpeg
        self.use_lighting = use_lighting
        self.p_downscale = probs["downscale"]
        self.p_blur = probs["blur"]
        self.p_occlusion = probs["occlusion"]
        self.p_jpeg = probs["jpeg"]
        self.p_lighting = probs["lighting"]

    def __call__(self, img):
        if self.use_lighting:
            img = random_lighting(img, prob=self.p_lighting)
        if self.use_downscale:
            img = random_downscale(img, prob=self.p_downscale)
        if self.use_blur:
            img = random_blur(img, prob=self.p_blur)
        if self.use_jpeg:
            img = random_jpeg(img, prob=self.p_jpeg)
        if self.use_occlusion:
            img = random_occlusion(img, prob=self.p_occlusion)
        return img

    @staticmethod
    def degrade_for_eval(img, severity="medium"):
        """评测用：确定性退化（固定随机种子），用于监控场景退化评测。"""
        rng = np.random.RandomState(42)
        np.random.seed(42)
        aug = SurveillanceAugment(severity=severity)
        return aug(img)