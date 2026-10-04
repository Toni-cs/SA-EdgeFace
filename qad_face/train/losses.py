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
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine * cosine + 1e-9)
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