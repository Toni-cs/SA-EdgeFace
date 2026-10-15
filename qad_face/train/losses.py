"""
训练 loss：ArcFace（人脸识别分类）+ 蒸馏 loss（C2）。

ArcFace: 人脸识别标准 margin loss。
DistillLoss: 特征级蒸馏（学生 embedding 拟合教师 embedding）+ 可选关系蒸馏。

对应论文：Section 3.5（C2 蒸馏目标）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace margin loss。"""

    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = float(torch.cos(torch.tensor(m)))
        self.sin_m = float(torch.sin(torch.tensor(m)))
        self.th = float(torch.cos(torch.tensor(3.14159265) - m))
        self.mm = self.sin_m * m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        embedding = embedding.float()
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight.float())).float()
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s
        loss = F.cross_entropy(output, label)
        return loss, output


class FeatureDistillLoss(nn.Module):
    """
    特征级蒸馏 loss：学生 embedding 拟合教师 embedding。

    支持 'mse' 与 'cosine' 两种模式。
    """

    def __init__(self, mode="cosine"):
        super().__init__()
        self.mode = mode

    def forward(self, student_emb, teacher_emb):
        student_emb = student_emb.float()
        if self.mode == "mse":
            return F.mse_loss(student_emb, teacher_emb.detach())
        s_norm = F.normalize(student_emb, p=2, dim=1)
        t_norm = F.normalize(teacher_emb.detach(), p=2, dim=1)
        return 1.0 - (s_norm * t_norm).sum(dim=1).mean()


class RelationDistillLoss(nn.Module):
    """关系蒸馏：保持样本间相似度结构（教师 vs 学生）。"""

    def forward(self, student_emb, teacher_emb):
        s_sim = self._pairwise_cos(student_emb)
        t_sim = self._pairwise_cos(teacher_emb.detach())
        return F.mse_loss(s_sim, t_sim)

    @staticmethod
    def _pairwise_cos(emb):
        emb = F.normalize(emb, p=2, dim=1)
        return emb @ emb.t()


class DistillTotalLoss(nn.Module):
    """
    蒸馏总 loss = w_arc * ArcFace + w_feat * FeatureDistill + w_rel * RelationDistill。

    消融：通过权重为 0 关闭某项。
    """

    def __init__(self, arcface, w_arc=1.0, w_feat=0.1, w_rel=0.0, feat_mode="cosine"):
        super().__init__()
        self.arcface = arcface
        self.w_arc = w_arc
        self.w_feat = w_feat
        self.w_rel = w_rel
        self.feat_loss = FeatureDistillLoss(mode=feat_mode)
        self.rel_loss = RelationDistillLoss()

    def forward(self, student_emb, teacher_emb, label):
        loss_arc, logits = self.arcface(student_emb, label)
        total = self.w_arc * loss_arc
        logs = {"loss_arc": loss_arc.item()}
        if self.w_feat > 0 and teacher_emb is not None:
            loss_feat = self.feat_loss(student_emb, teacher_emb)
            total = total + self.w_feat * loss_feat
            logs["loss_feat"] = loss_feat.item()
        if self.w_rel > 0 and teacher_emb is not None and student_emb.size(0) > 1:
            loss_rel = self.rel_loss(student_emb, teacher_emb)
            total = total + self.w_rel * loss_rel
            logs["loss_rel"] = loss_rel.item()
        logs["loss_total"] = total.item()
        return total, logits, logs


# ============================================================================
# T-R2 / T-R3 实现（prereg_v2_matrix §6 细则 1-3，P4 2026-10-14 复签判决 2-②）
# 因子空间 = {d0..d4}；本表之外的组合不属注册空间，不得产生。
# ============================================================================

DISTILL_TYPES = {
    # 细则 1：d1 λ = λ_max 全局（与 D3 同 λ_max——H4 隔离「关系结构」变量必须锁 λ）
    # 细则 2：d2 = RKD-D only（rkd_d=1.0 / rkd_a=0.0），registered-secondary
    # 细则 3：d3 = RKD-D + RKD-A 等权（1.0/1.0）经 λ_max 全局缩放；
    #         d4 = 同 d3 结构 + 逐样本 λ_i 调制
    "d0": dict(point=False, rkd_d=0.0, rkd_a=0.0, cond=False),
    "d1": dict(point=True,  rkd_d=0.0, rkd_a=0.0, cond=False),
    "d2": dict(point=False, rkd_d=1.0, rkd_a=0.0, cond=False),
    "d3": dict(point=False, rkd_d=1.0, rkd_a=1.0, cond=False),
    "d4": dict(point=False, rkd_d=1.0, rkd_a=1.0, cond=True),
}


def quality_cond_g(emb, alpha=1.0):
    """T-R2：g_i = sigmoid(α·(‖f_i‖ − mean_batch(‖f‖)))，f_i = 学生当前 batch 嵌入。

    stop-gradient 只读（T-R2 人工必查点）：范数在 no_grad 下以 float32 计算——
    梯度若能经范数项回传，模型会人为膨胀特征范数以放大 λ_i，机制即被破坏。
    g ∈ (0,1)；λ_i = λ_max · g_i。
    """
    with torch.no_grad():
        norms = emb.detach().float().norm(p=2, dim=1)
        g = torch.sigmoid(alpha * (norms - norms.mean()))
    return g


class RKDLoss(nn.Module):
    """T-R3：RKD-D（距离项）/ RKD-A（角度项），独立权重开关。

    RKD-D：batch 内样本对欧氏距离（B×B 上三角），师生**各自**按 batch 均值归一化
      （消除尺度差；除零保护 clamp_min(1e-12)——T-R3 人工必查点），逐对 smooth-L1。
    RKD-A：三元组（i,j,k）角度 ∠(i,j,k) = arccos((e_ij·e_ik)/(|e_ij||e_ik|))，
      e_ij = f_i − f_j（卡公式；角点 = i）；cos 夹紧 [−1+1e-6, 1−1e-6] 防 arccos 端点
      梯度爆炸；三元组采样上限 n_triplets（卡：如 4096 个/step，内存预算内）。
    权重注入：w_pair/w_trip 逐元素外部权重（d4 的 λ_i 调制），加权均值、
      权重和 clamp_min(1e-12) 保护。
    数值：全程 float32 + autocast 关闭（关系量对精度敏感）。
    """

    def __init__(self, rkd_d_weight=1.0, rkd_a_weight=1.0, n_triplets=4096, seed=0):
        super().__init__()
        self.rkd_d_weight = float(rkd_d_weight)
        self.rkd_a_weight = float(rkd_a_weight)
        self.n_triplets = int(n_triplets)
        self._gen = torch.Generator().manual_seed(int(seed))

    @staticmethod
    def distance_matrix(z):
        """(B,B) 欧氏距离矩阵，float32。"""
        return torch.cdist(z.float(), z.float())

    @staticmethod
    def angles(z, i, j, k):
        """三元组角度（弧度），公式与单测手算一致：∠(i,j,k) = arccos((e_ij·e_ik)/(|e_ij||e_ik|))。"""
        z = z.float()
        e_ij = z[i] - z[j]
        e_ik = z[i] - z[k]
        dot = (e_ij * e_ik).sum(dim=1)
        denom = e_ij.norm(dim=1) * e_ik.norm(dim=1)
        cos = (dot / denom.clamp_min(1e-12)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cos)

    def sample_triples(self, B, device):
        """无放回三元组采样（i≠j、i≠k、j≠k），上限 min(n_triplets, B(B-1)(B-2))。

        专用 CPU Generator（__init__ 播种）——与全局 RNG 解耦，同 seed 同代码路径逐位复现。
        """
        n = min(self.n_triplets, B * (B - 1) * (B - 2))
        i = torch.randint(0, B, (n,), generator=self._gen)
        j = torch.randint(0, B - 1, (n,), generator=self._gen)
        j = j + (j >= i).long()
        a = torch.minimum(i, j)
        b = torch.maximum(i, j)
        k = torch.randint(0, B - 2, (n,), generator=self._gen)
        k = k + (k >= a).long()      # 串行调整：第二次须基于更新后的值（并行会在 r=b-1 撞 b）
        k = k + (k >= b).long()
        return i.to(device), j.to(device), k.to(device)

    @staticmethod
    def pair_weights(g):
        """逐对权重 (g_i+g_j)/2，顺序 = 上三角 (i<j) 行主序，与 distance_matrix 取向一致。"""
        B = g.size(0)
        iu = torch.triu_indices(B, B, offset=1, device=g.device)
        return (g[iu[0]] + g[iu[1]]) / 2.0

    def forward(self, student_emb, teacher_emb, w_pair=None, w_trip=None, triples=None):
        with torch.amp.autocast("cuda", enabled=False):
            zs = student_emb.float()
            zt = teacher_emb.detach().float()
            B = zs.size(0)
            logs = {}
            loss = zs.new_zeros(())
            if self.rkd_d_weight > 0:
                iu = torch.triu_indices(B, B, offset=1, device=zs.device)
                ds = self.distance_matrix(zs)[iu[0], iu[1]]
                dt = self.distance_matrix(zt)[iu[0], iu[1]]
                ds = ds / ds.mean().clamp_min(1e-12)
                dt = dt / dt.mean().clamp_min(1e-12)
                l_elem = F.smooth_l1_loss(ds, dt, reduction="none")
                if w_pair is not None:
                    w = w_pair.float()
                    l_d = (l_elem * w).sum() / w.sum().clamp_min(1e-12)
                else:
                    l_d = l_elem.mean()
                loss = loss + self.rkd_d_weight * l_d
                logs["rkd_d"] = float(l_d)
            if self.rkd_a_weight > 0 and B >= 3:
                if triples is None:
                    triples = self.sample_triples(B, zs.device)
                i3, j3, k3 = triples
                ang_s = self.angles(zs, i3, j3, k3)
                ang_t = self.angles(zt.detach(), i3, j3, k3)
                l_elem = F.smooth_l1_loss(ang_s, ang_t, reduction="none")
                if w_trip is not None:
                    w = w_trip.float()
                    l_a = (l_elem * w).sum() / w.sum().clamp_min(1e-12)
                else:
                    l_a = l_elem.mean()
                loss = loss + self.rkd_a_weight * l_a
                logs["rkd_a"] = float(l_a)
            return loss, logs


class DistillV2Loss(nn.Module):
    """v2.0 注册矩阵蒸馏损失（prereg_v2_matrix §6 细则 1-3）。

    - d0：无蒸馏（仅 ArcFace）
    - d1：点对点 KD（cosine，沿用 FeatureDistillLoss），λ = λ_max 全局
    - d2：RKD-D only，经 λ_max 全局缩放（registered-secondary）
    - d3：RKD-D + RKD-A 等权（1.0/1.0），经 λ_max 全局缩放
    - d4：同 d3 结构 + 逐样本 λ_i 调制：g_i = quality_cond_g（stop-grad 只读），
      w_ij=(g_i+g_j)/2、w_ijk=(g_i+g_j+g_k)/3，加权均值

    返回 (loss, logits, logs) 与 DistillTotalLoss 同形；logs 附 g 分布
    （min/max/mean/p25/p75，T-R2 #4）与 rkd_d/rkd_a 分量。
    """

    def __init__(self, arcface, distill_type, lambda_max=0.1, alpha=1.0,
                 rkd_d_weight=None, rkd_a_weight=None, n_triplets=4096,
                 w_arc=1.0, feat_mode="cosine", seed=0):
        super().__init__()
        if distill_type not in DISTILL_TYPES:
            raise ValueError(f"distill_type {distill_type} 未知，choose from {list(DISTILL_TYPES)}")
        spec = DISTILL_TYPES[distill_type]
        self.distill_type = distill_type
        self.lambda_max = float(lambda_max)
        self.alpha = float(alpha)
        self.spec = spec
        self.arcface = arcface
        self.w_arc = w_arc
        self.feat_loss = FeatureDistillLoss(mode=feat_mode) if spec["point"] else None
        rkd_d = spec["rkd_d"] if rkd_d_weight is None else float(rkd_d_weight)
        rkd_a = spec["rkd_a"] if rkd_a_weight is None else float(rkd_a_weight)
        self.rkd = RKDLoss(rkd_d_weight=rkd_d, rkd_a_weight=rkd_a,
                           n_triplets=n_triplets, seed=seed) if (rkd_d > 0 or rkd_a > 0) else None

    def forward(self, student_emb, teacher_emb, label):
        loss_arc, logits = self.arcface(student_emb, label)
        total = self.w_arc * loss_arc
        logs = {"loss_arc": loss_arc.item()}
        if teacher_emb is not None:
            if self.feat_loss is not None:
                loss_feat = self.feat_loss(student_emb, teacher_emb)
                total = total + self.lambda_max * loss_feat
                logs["loss_feat"] = loss_feat.item()
            if self.rkd is not None and student_emb.size(0) > 2:
                w_pair = w_trip = None
                triples = None
                if self.spec["cond"]:
                    g = quality_cond_g(student_emb, alpha=self.alpha)
                    B = g.size(0)
                    iu = torch.triu_indices(B, B, offset=1, device=g.device)
                    w_pair = (g[iu[0]] + g[iu[1]]) / 2.0
                    triples = self.rkd.sample_triples(B, student_emb.device)
                    w_trip = (g[triples[0]] + g[triples[1]] + g[triples[2]]) / 3.0
                    logs.update({
                        "g_min": float(g.min()), "g_max": float(g.max()),
                        "g_mean": float(g.mean()),
                        "g_p25": float(g.quantile(0.25)), "g_p75": float(g.quantile(0.75)),
                    })
                loss_rkd, rkd_logs = self.rkd(student_emb, teacher_emb,
                                              w_pair=w_pair, w_trip=w_trip, triples=triples)
                total = total + self.lambda_max * loss_rkd
                logs.update(rkd_logs)
                logs["loss_rkd"] = float(loss_rkd)
        logs["loss_total"] = total.item()
        return total, logits, logs