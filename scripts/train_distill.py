"""
C2 知识蒸馏训练入口。

用法:
    python scripts/train_distill.py --data_root datasets/train_subset \
        --student_config sab_full --teacher insightface_buffalo_l \
        --epochs 20 --batch_size 64 --device cuda

    # 消融：关闭特征蒸馏
    python scripts/train_distill.py --data_root datasets/train_subset \
        --student_config sab_full --w_feat 0 --w_rel 0 --epochs 20
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qad_face.train.dataset import build_train_dataset
from qad_face.train.distill import DistillTrainer, precompute_teacher_features
from qad_face.models.recognizer import InsightFaceRecognizer
from qad_face.models.edgeface import DEFAULT_CONFIGS


class IndexedDataset(torch.utils.data.Dataset):
    """包装 FaceClassDataset，额外返回 sample index（用于教师特征索引）。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, idx


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--data_root", type=str, required=True, help="训练数据根目录(identity folders)")
    parser.add_argument("--student_config", type=str, default="sab_full",
                        choices=list(DEFAULT_CONFIGS.keys()), help="学生模型配置")
    parser.add_argument("--teacher", type=str, default="insightface_buffalo_l",
                        help="教师模型: insightface_buffalo_l / buffalo_s / buffalo_m")
    parser.add_argument("--severity", type=str, default="medium",
                        choices=["none", "light", "medium", "heavy"], help="C3 监控退化增强强度")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w_arc", type=float, default=1.0)
    parser.add_argument("--w_feat", type=float, default=0.5)
    parser.add_argument("--w_rel", type=float, default=0.1)
    parser.add_argument("--no_teacher", action="store_true", help="无教师模式（纯 ArcFace，蒸馏权重归零）")
    parser.add_argument("--feat_mode", type=str, default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--use_aug", action="store_true", default=True)
    parser.add_argument("--no_aug", action="store_true", help="关闭 C3 增强")
    parser.add_argument("--cache_path", type=str, default=None, help="教师特征缓存路径")
    parser.add_argument("--save_dir", type=str, default="weights/distill")
    parser.add_argument("--save_name", type=str, default="student")
    parser.add_argument("--max_identities", type=int, default=None, help="限制身份数(调试用)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("[WARN] CUDA 不可用，回退到 CPU")

    print(f"[1/4] 构建训练数据集 (severity={args.severity})")
    use_aug = args.use_aug and not args.no_aug
    base = build_train_dataset(
        args.data_root, severity=args.severity, use_aug=use_aug,
        max_identities=args.max_identities,
    )
    dataset = IndexedDataset(base)
    print(f"  样本数: {len(dataset)}, 身份数: {base.num_classes}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    use_teacher = not args.no_teacher
    if use_teacher:
        print(f"[2/4] 初始化教师模型: {args.teacher}")
        teacher_name = args.teacher.replace("insightface_", "")
        teacher = InsightFaceRecognizer(model_name=teacher_name)

        cache = args.cache_path or os.path.join(args.save_dir, f"teacher_{teacher_name}.npy")
        print(f"[3/4] 预计算/加载教师特征 (cache={cache})")
        teacher_feats = precompute_teacher_features(base, teacher, cache_path=cache)
    else:
        print("[2/4] 跳过教师（--no_teacher，纯 ArcFace 模式）")
        print("[3/4] 跳过教师特征预计算")
        teacher_feats = torch.zeros(len(base), 512)

    print(f"[4/4] 开始蒸馏训练: student={args.student_config}")
    trainer = DistillTrainer(
        student_config=DEFAULT_CONFIGS[args.student_config],
        num_classes=base.num_classes, device=device,
        w_arc=args.w_arc,
        w_feat=args.w_feat if use_teacher else 0.0,
        w_rel=args.w_rel if use_teacher else 0.0,
        feat_mode=args.feat_mode,
    )
    trainer.train(
        dataloader, teacher_feats, epochs=args.epochs, lr=args.lr,
        save_dir=args.save_dir, save_name=args.save_name,
        use_amp=False,
    )


if __name__ == "__main__":
    main()