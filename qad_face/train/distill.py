"""
C2 知识蒸馏训练器：教师（InsightFace buffalo_l）→ 学生（改进 EdgeFace）。

策略：预计算教师特征缓存（每张 aligned face 过一次教师得 512 维 embedding），
训练时学生拟合缓存特征 + 自身 ArcFace 分类，避免在线跑教师。

总 loss = w_arc * ArcFace + w_feat * FeatureDistill + w_rel * RelationDistill
消融：通过权重为 0 关闭某项。

对应论文：Section 3.5（C2）+ Section 4.2（训练设置）。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.edgeface import build_edgeface
from .losses import ArcFaceLoss, DistillTotalLoss
from .dataset import build_train_dataset


def precompute_teacher_features(dataset, recognizer, cache_path=None, batch_size=64, device="cpu"):
    """
    预计算教师特征。recognizer 需有 get_feature(img_ndarray)->embedding。

    若 cache_path 存在则直接加载；否则计算并保存。
    """
    if cache_path and os.path.exists(cache_path):
        feats = np.load(cache_path)
        print(f"[Teacher] loaded cached features: {feats.shape} from {cache_path}")
        return torch.from_numpy(feats).float()

    print(f"[Teacher] precomputing features for {len(dataset)} samples...")
    feats = np.zeros((len(dataset), 512), dtype=np.float32)
    import cv2
    for i in range(len(dataset)):
        img = cv2.imread(dataset.paths[i])
        if img is None:
            continue
        emb = recognizer.get_feature(img)
        if emb is not None:
            feats[i] = emb
        if (i + 1) % 500 == 0:
            print(f"  [Teacher] {i+1}/{len(dataset)}")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, feats)
        print(f"[Teacher] cached features saved to {cache_path}")
    return torch.from_numpy(feats).float()


class DistillTrainer:
    """知识蒸馏训练器。"""

    def __init__(self, student_config, num_classes, device="cuda",
                 w_arc=1.0, w_feat=0.1, w_rel=0.0, feat_mode="cosine",
                 arc_s=30.0, arc_m=0.3, emb_dim=512, seed=None):
        """
        seed: 若提供，则在构建模型权重之前设置全局随机种子，保证初始化可复现。
              不传 seed 时保持 PyTorch 默认（不可复现，但兼容旧调用）。
        """
        import random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.device = device
        self.student = build_edgeface(student_config).to(device)
        self.arcface = ArcFaceLoss(emb_dim, num_classes, s=arc_s, m=arc_m).to(device)
        self.criterion = DistillTotalLoss(
            self.arcface, w_arc=w_arc, w_feat=w_feat, w_rel=w_rel, feat_mode=feat_mode
        ).to(device)
        self.emb_dim = emb_dim
        self.eval_history = []

    def train(self, dataloader, teacher_features, epochs=20, lr=1e-3,
              weight_decay=5e-4, save_dir="weights/distill", save_name="student",
              log_every=50, use_amp=True, grad_clip=10.0, resume_from=None,
              warmup_epochs=2, optimizer_type="sgd", distill_start_epoch=0, seed=None,
              eval_fn=None):
        """
        蒸馏训练。dataloader 需返回 (imgs, label, sample_idx)，
        teacher_features[sample_idx] 为对应教师 embedding。

        Args:
            grad_clip: 梯度范数裁剪上限，防止梯度爆炸
            resume_from: checkpoint 路径，从该 checkpoint 恢复训练（含 epoch 信息）
            warmup_epochs: 前 N 个 epoch 线性 warmup 学习率
            optimizer_type: "sgd" (momentum=0.9) 或 "adamw"
            distill_start_epoch: 前 N 个 epoch 只用 ArcFace，之后再加蒸馏（避免早期梯度冲突）
            seed: 仅影响 DataLoader shuffle / 后续随机；模型初始化复现需在 __init__ 传 seed
            eval_fn: callable(model, device) -> dict，每 2 个 epoch 调用一次（如 AUC/Acc）
        """
        import random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        import math
        os.makedirs(save_dir, exist_ok=True)
        self.student.train()
        params = list(self.student.parameters()) + list(self.arcface.parameters())
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            if epoch < epochs * 2 // 3:
                return 1.0
            if epoch < epochs * 5 // 6:
                return 0.1
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and self.device != "cpu"))

        teacher_features = teacher_features.to(self.device) if teacher_features is not None else None

        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            self.student.load_state_dict(ckpt["student"])
            self.arcface.load_state_dict(ckpt["arcface"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"[resume] loaded {resume_from}, starting from epoch {start_epoch}")

        global_step = 0
        nan_skip_count = 0
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        ran_epochs = 0
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            use_distill = teacher_features is not None and epoch >= distill_start_epoch
            running = 0.0
            n = 0
            for imgs, labels, idxs in dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                t_feat = teacher_features[idxs.to(self.device)] if use_distill else None

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(use_amp and self.device != "cpu")):
                    emb = self.student(imgs)
                    loss, _, logs = self.criterion(emb, t_feat, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_skip_count += 1
                    if nan_skip_count <= 3:
                        print(f"  [WARN] NaN/Inf loss at step {global_step}, skipping")
                    global_step += 1
                    total_steps = (steps_per_epoch or (global_step + 1)) * (epochs - start_epoch)
                    if nan_skip_count > max(10, total_steps * 0.01):
                        print(f"[FATAL] NaN/Inf step 超阈值 ({nan_skip_count} > 1% steps)，"
                              f"判定训练发散，终止并删除坏权重")
                        bad = os.path.join(save_dir, f"{save_name}_final.pt")
                        if os.path.exists(bad):
                            os.remove(bad)
                        raise RuntimeError(f"training diverged: {nan_skip_count} NaN steps")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * imgs.size(0)
                n += imgs.size(0)
                global_step += 1
                if global_step % log_every == 0:
                    extras = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "loss_total")
                    print(f"  [step {global_step}] total={loss.item():.4f} {extras}")
            scheduler.step()
            avg = running / max(n, 1)
            print(f"[epoch {epoch+1}/{epochs}] avg_loss={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} time={time.time()-t0:.1f}s"
                  f"{f' nan_skipped={nan_skip_count}' if nan_skip_count else ''}")
            self.eval_history.append({"epoch": epoch + 1, "avg_loss": avg})
            if eval_fn is not None and (epoch + 1 - start_epoch) % 2 == 0:
                try:
                    m = eval_fn(self.student, self.device)
                    if isinstance(m, dict):
                        self.eval_history[-1].update(m)
                        print(f"  [eval ep{epoch+1}] AUC={m.get('auc', float('nan')):.4f} "
                              f"Acc={m.get('acc', float('nan'))*100:.2f}%")
                except Exception as e:
                    print(f"  [eval warn] eval_fn 失败: {e}")
            self.save(os.path.join(save_dir, f"{save_name}_ep{epoch+1}.pt"), optimizer, scheduler, epoch + 1, avg)
            ran_epochs += 1
        if ran_epochs < epochs - start_epoch:
            raise RuntimeError(f"训练在 ep{start_epoch + ran_epochs} 提前终止")
        self.save(os.path.join(save_dir, f"{save_name}_final.pt"), optimizer, scheduler, epochs, avg)
        print(f"[done] final model saved to {save_dir}/{save_name}_final.pt")

    def save(self, path, optimizer=None, scheduler=None, epoch=None, train_loss=None):
        if optimizer is not None and scheduler is not None:
            has_nan = any(torch.isnan(p).any().item()
                          for p in self.student.parameters() if p.requires_grad)
            if has_nan:
                print(f"[WARN] student 权重含 NaN，禁止覆盖 {path}")
                return
        ckpt = {
            "student": self.student.state_dict(),
            "arcface": self.arcface.state_dict(),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            ckpt["epoch"] = epoch
        if train_loss is not None:
            ckpt["train_loss"] = train_loss
        ckpt["eval_history"] = self.eval_history
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student"])
        self.arcface.load_state_dict(ckpt["arcface"])