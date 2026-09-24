"""
冒烟测试：验证人脸识别 pipeline 是否正常工作。
用两张本地图片测试检测、特征提取、相似度计算。

用法:
    python scripts/smoke_test.py --img1 face1.jpg --img2 face2.jpg
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.models.recognizer import InsightFaceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Smoke Test")
    parser.add_argument("--img1", type=str, required=True, help="第一张人脸图片路径")
    parser.add_argument("--img2", type=str, required=True, help="第二张人脸图片路径")
    parser.add_argument("--model", type=str, default="buffalo_s",
                        help="insightface 模型名")
    args = parser.parse_args()

    if not os.path.exists(args.img1):
        print(f"[ERROR] 文件不存在: {args.img1}")
        return
    if not os.path.exists(args.img2):
        print(f"[ERROR] 文件不存在: {args.img2}")
        return

    print(f"[INFO] 加载模型: {args.model}")
    t0 = time.time()
    recognizer = InsightFaceRecognizer(model_name=args.model)
    print(f"[INFO] 模型加载耗时: {time.time()-t0:.2f}s")

    # 检测第一张图
    print(f"\n[INFO] 处理图片1: {args.img1}")
    t0 = time.time()
    faces1 = recognizer.get_all_faces(args.img1)
    print(f"[INFO] 检测到 {len(faces1)} 张人脸, 耗时: {time.time()-t0:.3f}s")
    if len(faces1) == 0:
        print("[ERROR] 未检测到人脸！")
        return
    feat1 = faces1[0]["embedding"]
    print(f"[INFO] 特征维度: {feat1.shape}, 范数: {np.linalg.norm(feat1):.4f}")

    # 检测第二张图
    print(f"\n[INFO] 处理图片2: {args.img2}")
    t0 = time.time()
    faces2 = recognizer.get_all_faces(args.img2)
    print(f"[INFO] 检测到 {len(faces2)} 张人脸, 耗时: {time.time()-t0:.3f}s")
    if len(faces2) == 0:
        print("[ERROR] 未检测到人脸！")
        return
    feat2 = faces2[0]["embedding"]

    # 计算相似度
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    print(f"\n{'='*50}")
    print(f"  余弦相似度: {sim:.4f}")
    print(f"  判断: {'同一人 (相似度>0.3)' if sim > 0.3 else '不同人 (相似度<=0.3)'}")
    print(f"{'='*50}")
    print("\n[OK] Smoke test passed! Pipeline 工作正常。")


if __name__ == "__main__":
    main()
