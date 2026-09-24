"""
ShuffleNetV2 骨干网络（EdgeFace 的基座），并集成 C1 SAB 注意力增强。

SAB 插入位置可配置（默认深层 stage3/stage4 每个 unit 后），支持消融：
- sab_stages: 在哪些 stage 插入 SAB，如 (3,4)
- sab_config: SAB 子分支开关 dict
- use_sab: 总开关

对应论文：Section 3.1（骨干）+ Section 3.2（C1 集成）。
"""

import torch
import torch.nn as nn
from .attention import build_sab


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


class InvertedResidual(nn.Module):
    """ShuffleNetV2 基本单元。"""

    def __init__(self, inp, oup, stride):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        branch_features = oup // 2

        if stride == 1:
            assert inp == oup
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride == 2 else branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 骨干，可集成 SAB。

    Args:
        width_mult: 宽度倍率，0.5 / 1.0 / 1.5 / 2.0
        sab_stages: 在哪些 stage 的每个 unit 后插入 SAB，如 (3, 4)
        sab_config: SAB 子分支配置 dict（消融）
        use_sab: 是否启用 SAB
    """

    def __init__(self, width_mult=1.0, sab_stages=(3, 4), sab_config=None, use_sab=True):
        super().__init__()
        stages_repeats = {0.5: (4, 8, 4), 1.0: (4, 8, 4), 1.5: (7, 13, 7), 2.0: (7, 13, 7)}
        stages_out_channels = {
            0.5: (24, 48, 96, 192, 1024),
            1.0: (24, 116, 232, 464, 1024),
            1.5: (24, 176, 352, 704, 1024),
            2.0: (24, 244, 488, 976, 2048),
        }
        if width_mult not in stages_repeats:
            raise ValueError(f"width_mult {width_mult} not supported, choose from {list(stages_repeats)}")

        repeats = stages_repeats[width_mult]
        out_channels = stages_out_channels[width_mult]
        self.use_sab = use_sab
        self.sab_stages = set(sab_stages) if use_sab else set()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(repeats[0], out_channels[0], out_channels[1], 2, sab_config)
        self.stage3 = self._make_stage(repeats[1], out_channels[1], out_channels[2], 3, sab_config)
        self.stage4 = self._make_stage(repeats[2], out_channels[2], out_channels[3], 4, sab_config)

        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels[4]

    def _make_stage(self, repeat, inp, oup, stage_idx, sab_config):
        seq = nn.Sequential()
        inp_i = inp
        for i in range(repeat):
            stride = 2 if i == 0 else 1
            seq.add_module(f"unit{i}", InvertedResidual(inp_i, oup, stride=stride))
            inp_i = oup
            if stage_idx in self.sab_stages and i == repeat - 1:
                seq.add_module(f"sab_last", build_sab(oup, sab_config))
        return seq

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x
