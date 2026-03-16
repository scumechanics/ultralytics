import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 激活函数
# ─────────────────────────────────────────────

class Swish(nn.Module):
    """原论文使用的激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)


def build_act(act: str) -> nn.Module:
    """
    支持的激活函数:
        swish / silu  - 原论文默认，平滑非单调
        gelu          - Transformer 常用，效果接近 swish
        mish          - 比 swish 更平滑，梯度更稳定
        relu          - 最轻量
    """
    act = act.lower()
    mapping = {
        'swish': Swish,
        'silu':  nn.SiLU,
        'gelu':  nn.GELU,
        'mish':  nn.Mish,
        'relu':  nn.ReLU,
    }
    if act not in mapping:
        raise ValueError(f"[BiFPN] 不支持的激活函数: '{act}'，可选: {list(mapping.keys())}")
    return mapping[act]()


# ─────────────────────────────────────────────
# 融合策略 1: 标量加权融合（原始 BiFPN）
# ─────────────────────────────────────────────

class ScalarWeightedFusion(nn.Module):
    """
    每个输入分支分配一个可学习标量权重，
    经激活归一化后加权求和。
    对应原论文的 Fast Normalized Fusion。
    """
    def __init__(self, length: int, act: str = 'swish'):
        super().__init__()
        # 每个输入分支一个标量权重，初始化为 1
        self.weight = nn.Parameter(
            torch.ones(length, dtype=torch.float32), requires_grad=True
        )
        self.act = build_act(act)
        self.epsilon = 1e-4

    def forward(self, x: list) -> torch.Tensor:
        # 快速归一化: w_i / (sum(act(w)) + eps)
        w = self.act(self.weight)
        w = w / (w.sum() + self.epsilon)                          # [length]
        weighted = [w[i] * x[i] for i in range(len(x))]
        return torch.stack(weighted, dim=0).sum(dim=0)            # [B, C, H, W]


# ─────────────────────────────────────────────
# 融合策略 2: 通道级注意力融合（SE-style）
# ─────────────────────────────────────────────

class ChannelAttentionFusion(nn.Module):
    """
    对每个输入分支独立做 SE（Squeeze-and-Excitation）通道注意力，
    得到 [B, C, 1, 1] 的通道权重后与原特征相乘，最后求和。
    相比标量权重，能感知每个通道的重要性差异。
    """
    def __init__(self, length: int, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        # 每个输入分支独立的 SE 模块
        self.se_list = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # [B, C, 1, 1]
                nn.Flatten(),              # [B, C]
                nn.Linear(channels, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels),
                nn.Sigmoid(),              # 输出 [B, C]，值域 (0,1)
            )
            for _ in range(length)
        ])

    def forward(self, x: list) -> torch.Tensor:
        attended = []
        for i, feat in enumerate(x):
            B, C, H, W = feat.shape
            # SE 输出通道权重，reshape 为 [B, C, 1, 1] 后与特征图相乘
            attn = self.se_list[i](feat).view(B, C, 1, 1)
            attended.append(feat * attn)
        return torch.stack(attended, dim=0).sum(dim=0)            # [B, C, H, W]


# ─────────────────────────────────────────────
# 单层 BiFPN（内部使用）
# ─────────────────────────────────────────────

class BiFPN_Layer(nn.Module):
    """
    单层融合，根据 use_attention 选择融合策略。
    """
    def __init__(
        self,
        length: int,
        channels: int = 256,
        act: str = 'swish',
        use_attention: bool = False,
        reduction: int = 8,
    ):
        super().__init__()
        if use_attention:
            self.fusion = ChannelAttentionFusion(length, channels, reduction)
        else:
            self.fusion = ScalarWeightedFusion(length, act)

    def forward(self, x: list) -> torch.Tensor:
        return self.fusion(x)


# ─────────────────────────────────────────────
# 对外暴露的主模块 Bi_FPN
# ─────────────────────────────────────────────

class Bi_FPN(nn.Module):
    """
    可配置的双向特征金字塔融合模块。

    Args:
        length       (int):  输入分支数量，由 yaml 的 from 字段自动推断
        channels     (int):  输入特征图的通道数，由 tasks.py 自动传入
        num_layers   (int):  融合层重复次数，默认 1
        act          (str):  激活函数，默认 'swish'，仅标量模式有效
        use_attention(bool): True 使用通道注意力，False 使用标量加权
        reduction    (int):  SE 模块的通道压缩比，默认 8，仅注意力模式有效

    yaml 示例:
        # 标量加权，1层，swish
        - [[-1, 12], 1, Bi_FPN, []]

        # 标量加权，2层，gelu
        - [[-1, 12], 1, Bi_FPN, [2, 'gelu']]

        # 通道注意力，2层，reduction=8
        - [[-1, 12], 1, Bi_FPN, [2, 'swish', True, 8]]
    """

    def __init__(
        self,
        length: int,
        channels: int = 256,
        num_layers: int = 1,
        act: str = 'swish',
        use_attention: bool = False,
        reduction: int = 8,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPN_Layer(length, channels, act, use_attention, reduction)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: list of Tensor [B, C, H, W]，长度等于 length
        每层融合后，将输出广播回 list 供下一层使用
        """
        out = x if isinstance(x, list) else [x]
        for layer in self.layers:
            fused = layer(out)
            out = [fused] * len(out)   # 广播给下一层
        return fused
