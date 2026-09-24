"""
数据集对齐预处理：检测 + 对齐人脸，输出 112x112 aligned identity folders。

训练图、教师特征、评测输入必须同为对齐人脸，三方分布一致。

检测器可选：
  - insightface: buffalo 精确检测+对齐（需已下载模型）
  - haar: OpenCV 内置 Haar 级联（零依赖，快速验证用）

用法:
    python scripts/align_dataset.py --src datasets/train_mini_raw --dst datasets/train_mini
    python scripts/align_dataset.py --src ... --dst ... --detector haar
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.utils import imread_cn, imwrite_cn, load_cascade_ascii

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class HaarAligner:
    """OpenCV Haar 级联检测 + bbox 裁剪对齐（无关键点，快速验证用）。"""

    def __init__(self):
        xml_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = load_cascade_ascii(xml_path)
        if self.cascade.empty():
            raise RuntimeError("Haar cascade 加载失败")

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        best, best_area = None, 0
        for (x, y, w, h) in faces:
            if w * h > best_area:
                best_area = w * h
                best = (x, y, x + w, y + h)
        return [best] if best is not None else []


def main():
    parser = argparse.ArgumentParser(description="Align face dataset")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--detector", type=str, default="haar",
                        choices=["haar", "insightface"])
    parser.add_argument("--model", type=str, default="buffalo_s")
    args = parser.parse_args()

    if args.detector == "insightface":
        from qad_face.models.recognizer import InsightFaceRecognizer
        rec = InsightFaceRecognizer(model_name=args.model)

        def detect(img):
            return [max(rec.get_all_faces(img), key=lambda x: x["det_score"])["bbox"].astype(int)]
    else:
        haar = HaarAligner()

        def detect(img):
            return haar.detect(img)

    os.makedirs(args.dst, exist_ok=True)

    total, aligned, missed = 0, 0, 0
    for ident in sorted(os.listdir(args.src)):
        id_dir = os.path.join(args.src, ident)
        if not os.path.isdir(id_dir):
            continue
        out_dir = os.path.join(args.dst, ident)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(id_dir)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            total += 1
            img = imread_cn(os.path.join(id_dir, f))
            if img is None:
                missed += 1
                continue
            boxes = detect(img)
            if len(boxes) == 0:
                missed += 1
                continue
            x1, y1, x2, y2 = boxes[0]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                missed += 1
                continue
            crop = cv2.resize(crop, (112, 112))
            name, _ = os.path.splitext(f)
            imwrite_cn(os.path.join(out_dir, f"{name}.jpg"), crop,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            aligned += 1
        print(f"  {ident}: done")

    print(f"\n[done] total={total} aligned={aligned} missed={missed} ({missed/max(total,1)*100:.1f}%)")
    n_ids = len([d for d in os.listdir(args.dst) if os.path.isdir(os.path.join(args.dst, d))])
    print(f"[done] {n_ids} identities -> {args.dst}")


if __name__ == "__main__":
    main()