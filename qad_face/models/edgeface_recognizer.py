"""
改进 EdgeFace 的 Recognizer 封装：检测对齐用 insightface，识别用训练好的 EdgeFace。

兼容 qad_face.models.recognizer.BaseRecognizer 接口，可直接接入 eval_lfw.py 评测。
也支持加载 ONNX 导出的 EdgeFace（部署用）。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .recognizer import BaseRecognizer
from .edgeface import build_edgeface, DEFAULT_CONFIGS


class EdgeFaceRecognizer(BaseRecognizer):
    """检测对齐用 insightface，识别用 EdgeFace（PyTorch 权重）。"""

    def __init__(self, ckpt_path=None, config=None, config_name=None,
                 device="cpu", input_size=(112, 112), det_root="weights",
                 det_model="buffalo_s"):
        super().__init__(name="edgeface")
        self.input_size = input_size
        self.device = device

        if config_name is not None:
            config = DEFAULT_CONFIGS[config_name]
        self.model = build_edgeface(config)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt["student"] if "student" in ckpt else ckpt
            self.model.load_state_dict(state, strict=False)
            print(f"[EdgeFace] loaded checkpoint: {ckpt_path}")
        self.model.to(device).eval()

        from insightface.app import FaceAnalysis
        self.detector = FaceAnalysis(
            name=det_model, root=det_root, providers=["CPUExecutionProvider"]
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
            else:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                crop = img[y1:y2, x1:x2]
                aligned.append(cv2.resize(crop, self.input_size))
        return aligned

    def extract_feature(self, aligned_face):
        if aligned_face is None:
            return None
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, self.input_size)
        face = face.astype(np.float32) / 127.5 - 1.0
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.get_embedding(tensor)
        return emb.cpu().numpy().flatten()

    def get_feature(self, img):
        faces = self.detect_and_align(img)
        if len(faces) == 0:
            return None
        return self.extract_feature(faces[0])

    def get_all_faces(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        det_faces = self.detector.get(img)
        results = []
        for face in det_faces:
            crop = face.crop if (hasattr(face, "crop") and face.crop is not None) else None
            emb = self.extract_feature(crop) if crop is not None else None
            results.append({"bbox": face.bbox, "embedding": emb, "det_score": face.det_score})
        return results


class EdgeFaceONNXRecognizer(BaseRecognizer):
    """加载 ONNX 导出的 EdgeFace（部署/评测用，无需 PyTorch）。"""

    def __init__(self, onnx_path, input_size=(112, 112), det_root="weights",
                 det_model="buffalo_s", name="edgeface_onnx"):
        super().__init__(name=name)
        self.input_size = input_size
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        from insightface.app import FaceAnalysis
        self.detector = FaceAnalysis(
            name=det_model, root=det_root, providers=["CPUExecutionProvider"]
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
        if aligned_face is None:
            return None
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, self.input_size)
        face = face.astype(np.float32) / 127.5 - 1.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        feat = self.session.run([self.output_name], {self.input_name: face})[0]
        return feat.flatten()

    def get_feature(self, img):
        faces = self.detect_and_align(img)
        if len(faces) == 0:
            return None
        return self.extract_feature(faces[0])