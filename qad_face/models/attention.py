"""
C1 主创新：Surveillance Attention Block (SAB) —— 面向监控小人脸的轻量注意力增强模块。

设计动机：监控场景下人脸分辨率低、全局结构信息丢失，识别更依赖局部细节纹理。
SAB 在通道注意力与空间注意力之外，引入多尺度局部细节增强分支，三者均可独立开关以支持消融。

对应论文：Section 3.2 (C1)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECAChannelAttention(nn.Module):
    """ECA 风格通道注意力：用 1D 卷积替代 FC 降维，几乎零额外参数。"""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    """轻量空间注意力：深度可分离卷积生成空间权重图。"""

    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pwconv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pwconv(self.bn(self.dwconv(x)))
        return x * self.sigmoid(y)


class LocalDetailEnhance(nn.Module):
    """多尺度局部细节增强分支：3x3 与 5x5 深度卷积捕捉不同尺度局部纹理。"""

    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        d = self.dw3(x) + self.dw5(x)
        d = self.bn(self.pw(d))
        return d


class SurveillanceAttentionBlock(nn.Module):
    """
    SAB：面向监控场景的轻量注意力增强模块（论文 C1）。

    out = x * channel_att * spatial_att + alpha * local_detail

    Args:
        channels: 输入通道数
        use_channel: 通道注意力开关（消融）
        use_spatial: 空间注意力开关（消融）
        use_detail: 局部细节增强开关（消融）
        alpha_init: 局部细节分支的初始权重
    """

    def __init__(self, channels, use_channel=True, use_spatial=True,
                 use_detail=True, alpha_init=0.1):
        super().__init__()
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        self.use_detail = use_detail

        if use_channel:
            self.channel_att = ECAChannelAttention(channels)
        if use_spatial:
            self.spatial_att = SpatialAttention(channels)
        if use_detail:
            self.detail = LocalDetailEnhance(channels)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        out = x
        if self.use_channel:
            out = self.channel_att(out)
        if self.use_spatial:
            out = self.spatial_att(out)
        if self.use_detail:
            out = out + self.alpha * self.detail(x)
        return out


def build_sab(channels, config=None):
    """根据消融配置构建 SAB。config 为 dict 或 None。"""
    if config is None:
        return SurveillanceAttentionBlock(channels)
    return SurveillanceAttentionBlock(
        channels,
        use_channel=config.get("use_channel", True),
        use_spatial=config.get("use_spatial", True),
        use_detail=config.get("use_detail", True),
        alpha_init=config.get("alpha_init", 0.1),
    )