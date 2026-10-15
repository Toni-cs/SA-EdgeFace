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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.edgeface import build_edgeface
from .losses import ArcFaceLoss, DistillTotalLoss, DistillV2Loss
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
                 arc_s=30.0, arc_m=0.3, emb_dim=512, seed=None,
                 distill_type=None, lambda_max=0.1, alpha=1.0,
                 rkd_d_weight=None, rkd_a_weight=None, n_triplets=4096,
                 backbone_type="shufflenet"):
        """
        seed: 若提供，则在构建模型权重之前设置全局随机种子，保证初始化可复现。
              不传 seed 时保持 PyTorch 默认（不可复现，但兼容旧调用）。
        distill_type: v2.0 注册因子 d0..d4（None = 旧路径 DistillTotalLoss，历史 B/E 复现语义不变）。
        lambda_max/alpha: T-R2/T-R3 全局蒸馏尺度与质量条件化陡度（d1-d4 共用 λ_max 锁 λ）。
        backbone_type: "shufflenet"/"pconv"（T-R4）。
        """
        import random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.device = device
        self.distill_type = distill_type
        self.student = build_edgeface(student_config, backbone_type=backbone_type).to(device)
        self.arcface = ArcFaceLoss(emb_dim, num_classes, s=arc_s, m=arc_m).to(device)
        if distill_type is not None:
            self.criterion = DistillV2Loss(
                self.arcface, distill_type,
                lambda_max=lambda_max, alpha=alpha,
                rkd_d_weight=rkd_d_weight, rkd_a_weight=rkd_a_weight,
                n_triplets=n_triplets, w_arc=w_arc, feat_mode=feat_mode,
                seed=0 if seed is None else int(seed),
            ).to(device)
        else:
            self.criterion = DistillTotalLoss(
                self.arcface, w_arc=w_arc, w_feat=w_feat, w_rel=w_rel, feat_mode=feat_mode
            ).to(device)
        self.emb_dim = emb_dim
        self.eval_history = []

    def train(self, dataloader, teacher_features, epochs=20, lr=1e-3,
              weight_decay=5e-4, save_dir="weights/distill", save_name="student",
              log_every=50, use_amp=True, grad_clip=10.0, resume_from=None,
              warmup_epochs=2, optimizer_type="sgd", distill_start_epoch=0, seed=None,
              eval_fn=None, eval_every=0,
              dt_idxs=None, dt_labels=None, dt_every=500):
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
        eval_fn: callable(model, device) -> dict；eval_every>0 时每 eval_every 个 epoch 调用一次（默认 0=训练中不 eval，训练末由调用方自评一次）
        dt_idxs/dt_labels/dt_every: D(t) 插桩（D(t) 卡）——固定子集索引/标签 + 步频；
            每 dt_every step 真实前向计算 D(t)=E[‖f_S−f_T‖²]、D_norm(t)=E[1−cos]、
            类内/类间距离比 geo；曲线只来自真实日志，禁任何构造/平滑。
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
            g_ep = {"n": 0, "min": float("inf"), "max": 0.0, "sum": 0.0, "p25": 0.0, "p75": 0.0}
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
                if "g_mean" in logs:                       # d4：λ_i 分布逐 step 累积（T-R2 #4）
                    g_ep["n"] += 1
                    g_ep["min"] = min(g_ep["min"], logs["g_min"])
                    g_ep["max"] = max(g_ep["max"], logs["g_max"])
                    g_ep["sum"] += logs["g_mean"]
                    g_ep["p25"] += logs["g_p25"]
                    g_ep["p75"] += logs["g_p75"]
                if dt_idxs is not None and teacher_features is not None \
                        and global_step % dt_every == 0:
                    try:
                        self._dt_eval(dataloader, teacher_features, dt_idxs, dt_labels, global_step)
                    except Exception as e:
                        print(f"  [Dt warn] {e}")
                if global_step % log_every == 0:
                    extras = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "loss_total")
                    print(f"  [step {global_step}] total={loss.item():.4f} {extras}")
            scheduler.step()
            avg = running / max(n, 1)
            print(f"[epoch {epoch+1}/{epochs}] avg_loss={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} time={time.time()-t0:.1f}s"
                  f"{f' nan_skipped={nan_skip_count}' if nan_skip_count else ''}")
            if g_ep["n"] > 0:
                print(f"  [lambda_i ep{epoch+1}] min={g_ep['min']:.4f} max={g_ep['max']:.4f} "
                      f"mean={g_ep['sum']/g_ep['n']:.4f} p25={g_ep['p25']/g_ep['n']:.4f} "
                      f"p75={g_ep['p75']/g_ep['n']:.4f} (g = λ_i/λ_max，T-R2 #4)", flush=True)
            self.eval_history.append({"epoch": epoch + 1, "avg_loss": avg})
            if eval_fn is not None and eval_every > 0 and (epoch + 1 - start_epoch) % eval_every == 0:
                try:
                    m = eval_fn(self.student, self.device)
                    if isinstance(m, dict):
                        self.eval_history[-1].update(m)
                        print(f"  [eval ep{epoch+1}] AUC={m.get('auc', float('nan')):.4f} "
                              f"Acc={m.get('acc', float('nan'))*100:.2f}%")
                except Exception as e:
                    print(f"  [eval warn] eval_fn failed: {e}")
                finally:
                    self.student.train()
            self.save(os.path.join(save_dir, f"{save_name}_ep{epoch+1}.pt"), optimizer, scheduler, epoch + 1, avg)
            ran_epochs += 1
        if ran_epochs < epochs - start_epoch:
            raise RuntimeError(f"训练在 ep{start_epoch + ran_epochs} 提前终止")
        ok = self.save(os.path.join(save_dir, f"{save_name}_final.pt"), optimizer, scheduler, epochs, avg)
        if not ok:
            raise RuntimeError(f"final checkpoint 被健康门拒绝保存: {save_dir}/{save_name}_final.pt")
        print(f"[done] final model saved to {save_dir}/{save_name}_final.pt")

    @torch.no_grad()
    def _dt_eval(self, dataloader, teacher_features, dt_idxs, dt_labels, step):
        """D(t) 插桩：固定 2000 张子集（seed 固定，跨臂同集）真实前向。

        D(t) = E[‖f_S^t − f_T‖²]（原始嵌入）；D_norm(t) = E[1 − cos(f_S, f_T)]；
        geo = 类内/类间距离比（归一化嵌入上三角，按身份分组）。
        曲线唯一来源 = 本函数的真实计算 + 训练日志（D(t) 卡红线：禁构造/平滑）。
        """
        ds = dataloader.dataset
        self.student.eval()
        try:
            zs_list = []
            idxs_cpu = dt_idxs.cpu()
            for s0 in range(0, len(idxs_cpu), 256):
                batch = [ds[int(i)][0] for i in idxs_cpu[s0:s0 + 256]]
                imgs = torch.stack(batch).to(self.device)
                zs_list.append(self.student(imgs).float())
            zs = torch.cat(zs_list, 0)
            zt = teacher_features[dt_idxs.to(self.device)].float()
            d = float(((zs - zt) ** 2).sum(dim=1).mean())
            d_norm = float((1.0 - F.cosine_similarity(zs, zt, dim=1)).mean())
            zn = F.normalize(zs, dim=1)
            Bn = zn.size(0)
            iu = torch.triu_indices(Bn, Bn, offset=1, device=zn.device)
            dvec = torch.cdist(zn, zn)[iu[0], iu[1]]
            lab = dt_labels.to(self.device)
            same = (lab.view(-1, 1) == lab.view(1, -1))[iu[0], iu[1]]
            within = float(dvec[same].mean()) if bool(same.any()) else float("nan")
            between = float(dvec[~same].mean()) if bool((~same).any()) else float("nan")
            geo = within / between if between > 0 else float("nan")
            print(f"  [Dt step {step}] D={d:.4f} D_norm={d_norm:.4f} "
                  f"geo={geo:.4f} within={within:.4f} between={between:.4f}", flush=True)
        finally:
            self.student.train()

    def save(self, path, optimizer=None, scheduler=None, epoch=None, train_loss=None):
        if optimizer is not None and scheduler is not None:
            bad_nan = 0; abs_vals = []
            for p in self.student.parameters():
                if not p.requires_grad: continue
                t = p.detach().double()
                bad_nan += int(torch.isnan(t).sum()) + int(torch.isinf(t).sum())
                f = t[torch.isfinite(t)]
                if f.numel(): abs_vals.append(f.abs())
            if bad_nan:
                print(f"[WARN] student 权重含 {bad_nan} 个非有限值，禁止覆盖 {path}")
                return False
            if abs_vals:
                a = torch.cat(abs_vals); med = float(a.median())
                frac_dead = float((a < 1e-30).double().mean())
                print(f"[health] med|w|={med:.3e} frac(<1e-30)={frac_dead:.4f}")
                if med < 1e-8 or frac_dead > 0.01:
                    print(f"[WARN] 权重静默湮灭 (med|w|={med:.3e})，拒绝保存 {path}")
                    return False
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
        return True

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student"])
        self.arcface.load_state_dict(ckpt["arcface"])