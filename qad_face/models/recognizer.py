"""
人脸识别模型封装。

支持多种骨干网络的统一接口：
- InsightFaceRecognizer: 基于 insightface 的模型（检测+对齐+识别）
- EdgeFaceRecognizer: EdgeFace 模型（待接入）
- GhostFaceNetRecognizer: GhostFaceNet 模型（待接入）
"""

import os
import cv2
import numpy as np
import onnxruntime as ort


class BaseRecognizer:
    """识别器基类，定义统一接口。"""

    def __init__(self, name="base"):
        self.name = name
        self.input_size = (112, 112)

    def detect_and_align(self, img):
        """检测人脸并对齐，返回对齐后的人脸图像列表。"""
        raise NotImplementedError

    def extract_feature(self, aligned_face):
        """从对齐后的人脸提取特征向量。"""
        raise NotImplementedError

    def get_feature(self, img):
        """端到端：输入原图，返回第一张人脸的特征向量。"""
        faces = self.detect_and_align(img)
        if len(faces) == 0:
            return None
        return self.extract_feature(faces[0])


class InsightFaceRecognizer(BaseRecognizer):
    """
    基于 insightface 的识别器。
    使用 insightface 的检测+对齐，识别模型可替换。
    """

    def __init__(self, model_name="buffalo_s", root="weights", det_size=(640, 640)):
        super().__init__(name=f"insightface_{model_name}")
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model_name,
            root=root,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.input_size = (112, 112)

    def detect_and_align(self, img):
        """检测人脸，返回对齐后的 112x112 人脸图像列表。"""
        if isinstance(img, str):
            img = cv2.imread(img)
        faces = self.app.get(img)
        aligned = []
        for face in faces:
            # insightface 已经做了对齐，直接用裁剪后的人脸
            if hasattr(face, "crop") and face.crop is not None:
                aligned_face = face.crop
            else:
                # 回退：用 bbox 裁剪
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                aligned_face = img[y1:y2, x1:x2]
                aligned_face = cv2.resize(aligned_face, self.input_size)
            aligned.append(aligned_face)
        return aligned

    def extract_feature(self, aligned_face):
        """提取特征向量。insightface 的 face 对象已包含 embedding。"""
        # 如果传入的是已经检测过的 face 对象
        if hasattr(aligned_face, "embedding"):
            return aligned_face.embedding
        # 如果传入的是图像，重新检测
        if isinstance(aligned_face, np.ndarray) and aligned_face.ndim == 3:
            faces = self.app.get(aligned_face)
            if len(faces) > 0:
                return faces[0].embedding
        return None

    def get_all_faces(self, img):
        """返回所有人脸的 (bbox, embedding) 列表。"""
        if isinstance(img, str):
            img = cv2.imread(img)
        faces = self.app.get(img)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox,
                "embedding": face.embedding,
                "det_score": face.det_score,
            })
        return results


class ONNXRecognizer(BaseRecognizer):
    """
    通用 ONNX 识别器。
    用于加载 EdgeFace、GhostFaceNet 等导出的 ONNX 模型。
    检测和对齐使用 insightface，识别用指定的 ONNX 模型。
    """

    def __init__(self, onnx_path, name="onnx", input_size=(112, 112)):
        super().__init__(name=name)
        self.input_size = input_size
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # 检测对齐用 insightface
        from insightface.app import FaceAnalysis
        self.detector = FaceAnalysis(
            name="buffalo_s", root="weights",
            providers=["CPUExecutionProvider"],
        )
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

    def detect_and_align(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        faces = self.detector.get(img)
        aligned = []
        for face in faces:
            if hasattr(face, "crop") and face.crop is not None:
                aligned.append(face.crop)
        return aligned

    def extract_feature(self, aligned_face):
        """从对齐人脸提取特征。输入 BGR 图像。"""
        if aligned_face is None:
            return None
        # 预处理: BGR->RGB, resize, normalize
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, self.input_size)
        face = face.astype(np.float32) / 127.5 - 1.0
        face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # add batch

        feat = self.session.run(
            [self.output_name], {self.input_name: face}
        )[0]
        return feat.flatten()
