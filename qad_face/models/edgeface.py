"""
EdgeFace 完整模型：ShuffleNetV2 骨干 + 全局池化 + 识别头。

通过 build_edgeface(config) 工厂构建，config 控制消融开关：
- use_sab / sab_stages / sab_config：C1 注意力增强
- width_mult：骨干宽度
- emb_dim：嵌入维度（默认 512，与 InsightFace 对齐便于蒸馏）

对应论文：Section 3.3（整体架构）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shufflenetv2 import ShuffleNetV2


class EdgeFace(nn.Module):
    """EdgeFace：轻量人脸识别模型，可选集成 SAB（C1）。"""

    def __init__(self, width_mult=1.0, emb_dim=512, use_sab=True,
                 sab_stages=(3, 4), sab_config=None, dropout=0.0):
        super().__init__()
        self.backbone = ShuffleNetV2(
            width_mult=width_mult,
            sab_stages=sab_stages,
            sab_config=sab_config,
            use_sab=use_sab,
        )
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.backbone.out_channels, emb_dim, bias=False)

    def forward(self, x, normalize=False):
        feat = self.backbone(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        feat = self.dropout(feat)
        emb = self.fc(feat)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def get_embedding(self, x):
        return self.forward(x, normalize=True)


def build_edgeface(config=None):
    """
    根据配置 dict 构建 EdgeFace。

    config keys (均可选):
        width_mult: 0.5/1.0/1.5/2.0, 默认 1.0
        emb_dim: 默认 512
        use_sab: 默认 True
        sab_stages: 默认 (3,4)
        sab_config: SAB 子分支开关 {use_channel, use_spatial, use_detail, alpha_init}
        dropout: 默认 0.0
    """
    c = config or {}
    return EdgeFace(
        width_mult=c.get("width_mult", 1.0),
        emb_dim=c.get("emb_dim", 512),
        use_sab=c.get("use_sab", True),
        sab_stages=tuple(c.get("sab_stages", (3, 4))),
        sab_config=c.get("sab_config", None),
        dropout=c.get("dropout", 0.0),
    )


DEFAULT_CONFIGS = {
    "baseline": {"use_sab": False},
    "sab_full": {"use_sab": True, "sab_config": None},
    "sab_channel_only": {"use_sab": True, "sab_config": {"use_spatial": False, "use_detail": False}},
    "sab_spatial_only": {"use_sab": True, "sab_config": {"use_channel": False, "use_detail": False}},
    "sab_detail_only": {"use_sab": True, "sab_config": {"use_channel": False, "use_spatial": False}},
    "edgeface_x0_5": {"width_mult": 0.5, "use_sab": False},
    "edgeface_x0_5_sab": {"width_mult": 0.5, "use_sab": True},
    "x05_baseline": {"width_mult": 0.5, "use_sab": False},
    "x05_sab_channel": {"width_mult": 0.5, "use_sab": True, "sab_config": {"use_spatial": False, "use_detail": False}},
    "x05_sab_spatial": {"width_mult": 0.5, "use_sab": True, "sab_config": {"use_channel": False, "use_detail": False}},
    "x05_sab_full": {"width_mult": 0.5, "use_sab": True},
}


def build_by_name(name):
    """按预设名构建，用于实验脚本。"""
    if name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config {name}, choose from {list(DEFAULT_CONFIGS)}")
    return build_edgeface(DEFAULT_CONFIGS[name])