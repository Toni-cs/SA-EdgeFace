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
import torch.nn.functional as F
from .attention import build_sab


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


class PartialConvDW(nn.Module):
    """T-R4：FasterNet 风格 PConv，置于 InvertedResidual 的 DWConv 位置。

    结构（对齐 FasterNet CVPR 2023 的 PConv 语义，落在 ShuffleNetV2 DW 位）：
      - 前 1/4 通道（向下取整）走 3×3 常规卷积；
      - 其余通道 identity 直通；
      - 拼接后接 BN，随后进入**既有的** pointwise 卷积混合（branch 内 1×1 不变）。
    stride=2：identity 路径走 2×2 平均池化保持空间对齐——FasterNet 的下采样哲学
    （avgpool + partial conv）在 DW 位的折叠；恒等直通在跨步位置无定义，平均池化是
    保通道数的最近似。本件为 T-R4 注册实现形态，参数/FLOPs/延迟报告另见统计件。
    """

    def __init__(self, channels, stride=1, part=4):
        super().__init__()
        assert stride in (1, 2), f"stride {stride} 不支持（仅 1/2）"
        self.stride = stride
        self.part = part
        c = channels // part          # 向下取整：任意宽度表均可（0.5× 表全整除）
        self.keep = channels - c
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        trans = self.conv(x[:, :x.size(1) - self.keep])
        keep = x[:, x.size(1) - self.keep:]
        if self.stride == 2:
            # ceil_mode=True：输出尺寸与 Conv2d(k3,s2,p1) 逐例对齐（奇数边长 7→4 而非 3）
            keep = F.avg_pool2d(keep, kernel_size=2, stride=2, ceil_mode=True)
        return self.bn(torch.cat((trans, keep), dim=1))


class InvertedResidual(nn.Module):
    """ShuffleNetV2 基本单元。dw_impl: "std"（原 DWConv）/ "pconv"（T-R4 PConv 替换）。"""

    def __init__(self, inp, oup, stride, dw_impl="std"):
        super().__init__()
        assert stride in (1, 2)
        assert dw_impl in ("std", "pconv"), f"dw_impl {dw_impl} 未知"
        self.stride = stride
        branch_features = oup // 2

        def _dw(channels, st):
            if dw_impl == "pconv":
                return PartialConvDW(channels, stride=st)
            return nn.Conv2d(channels, channels, kernel_size=3, stride=st,
                             padding=1, groups=channels, bias=False)

        if stride == 1:
            assert inp == oup
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                _dw(inp, stride),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride == 2 else branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            _dw(branch_features, stride),
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
        backbone: "shufflenet"（标准 DWConv）/ "pconv"（T-R4 PConv 替换 DW 位）
    """

    def __init__(self, width_mult=1.0, sab_stages=(3, 4), sab_config=None, use_sab=True,
                 backbone="shufflenet"):
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
        if backbone not in ("shufflenet", "pconv"):
            raise ValueError(f"backbone {backbone} not supported, choose from ['shufflenet', 'pconv']")

        repeats = stages_repeats[width_mult]
        out_channels = stages_out_channels[width_mult]
        self.use_sab = use_sab
        self.backbone = backbone
        self.dw_impl = "pconv" if backbone == "pconv" else "std"
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
            seq.add_module(f"unit{i}", InvertedResidual(inp_i, oup, stride=stride,
                                                        dw_impl=self.dw_impl))
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
