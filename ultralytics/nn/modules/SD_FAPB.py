# ultralytics/nn/Addmodules/SD_FAPB.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['C2PSA_SDFAPB']


# ─── 基础工具 ────────────────────────────────────────────────

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True
                    else act if isinstance(act, nn.Module)
                    else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0,
                 dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


# ─── 核心模块：尺度解耦频率感知位置偏置 ──────────────────────

class ScaleDecoupledFAPB(nn.Module):
    """
    Scale-Decoupled Frequency-Aware Position Bias (SD-FAPB)

    动机：
        现有基于MLP的位置偏置（Swin V2等）让所有注意力头共享
        频率参数，忽略了多尺度目标检测中不同空间频率依赖的
        异质性需求：
          大目标 → 需要低频长程空间依赖
          小目标 → 需要高频短程局部依赖

    设计：
        1. 每个head拥有独立的可学习频率系数 freq_bands
        2. 差异化初始化引导不同head分化为低频头/高频头
        3. 每个head独立的MLP将频域特征映射为位置偏置
        4. 训练后可通过 get_freq_stats() 可视化频率分化现象

    与普通FAPB的关键区别：
        普通FAPB：所有head共享 freq_bands (num_freq,)
        SD-FAPB ：每个head独立 freq_bands (num_heads, num_freq)
    """

    # 预设频率初始化策略，引导不同head分化
    _FREQ_PRESETS = [
        (0.5,  2.0),   # 低频：大目标全局关系
        (1.0,  4.0),   # 中低频
        (2.0,  8.0),   # 中高频
        (4.0, 16.0),   # 高频：小目标局部细节
    ]

    def __init__(self, num_heads, hidden_dim=64, num_freq=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_freq = num_freq

        # 每个head独立的可学习频率系数，差异化初始化
        freq_rows = []
        for i in range(num_heads):
            lo, hi = self._FREQ_PRESETS[i % len(self._FREQ_PRESETS)]
            freq_rows.append(torch.linspace(lo, hi, num_freq))
        # shape: (num_heads, num_freq)
        self.freq_bands = nn.Parameter(torch.stack(freq_rows))

        # 每个head独立的MLP
        in_dim = 2 + 4 * num_freq
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_heads)
        ])

        self._cache = {}

    def _compute_bias(self, H, W, device):
        coords_h = torch.linspace(-1, 1, H, device=device)
        coords_w = torch.linspace(-1, 1, W, device=device)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        points = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)
        N = H * W
        rel = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 2)

        biases = []
        for i, mlp in enumerate(self.mlps):
            freq = self.freq_bands[i].to(device)              # (num_freq,)
            rel_h = rel[..., 0:1] * freq * torch.pi           # (N, N, num_freq)
            rel_w = rel[..., 1:2] * freq * torch.pi
            freq_feat = torch.cat([
                torch.sin(rel_h), torch.cos(rel_h),
                torch.sin(rel_w), torch.cos(rel_w),
            ], dim=-1)                                         # (N, N, 4*num_freq)
            feat = torch.cat([rel, freq_feat], dim=-1)         # (N, N, 2+4*num_freq)
            biases.append(mlp(feat))                           # (N, N, 1)

        # (num_heads, N, N)
        return torch.cat(biases, dim=-1).permute(2, 0, 1)

    def forward(self, H, W, device):
        if self.training:
            return self._compute_bias(H, W, device)
        key = (H, W)
        if key not in self._cache:
            self._cache[key] = self._compute_bias(H, W, device)
        return self._cache[key]

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._cache = {}
        return self

    def get_freq_stats(self):
        """训练后调用，返回每个head的频率统计，用于论文可视化"""
        with torch.no_grad():
            return {
                f'head{i}': {
                    'mean': self.freq_bands[i].mean().item(),
                    'min':  self.freq_bands[i].min().item(),
                    'max':  self.freq_bands[i].max().item(),
                }
                for i in range(self.num_heads)
            }


# ─── 级联组注意力（集成SD-FAPB）────────────────────────────

class SDFAPBCascadedGroupAttention(nn.Module):
    """
    集成SD-FAPB的级联组注意力
    多尺度kernel [3,5,7,9] 配合频率解耦，进一步增强多尺度感知
    """

    _MS_KERNELS = [3, 5, 7, 9]

    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)

        self.qkvs = nn.ModuleList([
            Conv2d_BN(dim // num_heads, self.key_dim * 2 + self.d)
            for _ in range(num_heads)
        ])

        self.dws = nn.ModuleList([
            Conv2d_BN(
                self.key_dim, self.key_dim,
                self._MS_KERNELS[i % len(self._MS_KERNELS)],
                1,
                self._MS_KERNELS[i % len(self._MS_KERNELS)] // 2,
                groups=self.key_dim
            )
            for i in range(num_heads)
        ])

        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0)
        )

        # SD-FAPB 替换原版静态 attention_bias
        self.sdfapb = ScaleDecoupledFAPB(
            num_heads, hidden_dim=64, num_freq=8
        )

    def forward(self, x):
        B, C, H, W = x.shape
        attn_bias = self.sdfapb(H, W, x.device)  # (num_heads, N, N)

        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]

        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split(
                [self.key_dim, self.key_dim, self.d], dim=1
            )
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

            attn = (q.transpose(-2, -1) @ k) * self.scale + attn_bias[i]
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)

        return self.proj(torch.cat(feats_out, 1))


class LocalWindowAttention_SDFAPB(nn.Module):
    def __init__(self, dim, num_heads=4, attn_ratio=4, window_resolution=7):
        super().__init__()
        self.window_resolution = window_resolution
        key_dim = max(dim // 16, 1)
        self.attn = SDFAPBCascadedGroupAttention(
            dim, key_dim, num_heads, attn_ratio
        )

    def forward(self, x):
        B, C, H, W = x.shape
        wr = self.window_resolution

        if H <= wr and W <= wr:
            return self.attn(x)

        x = x.permute(0, 2, 3, 1)
        pad_b = (wr - H % wr) % wr
        pad_r = (wr - W % wr) % wr
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // wr, pW // wr

        x = (x.view(B, nH, wr, nW, wr, C)
              .transpose(2, 3)
              .reshape(B * nH * nW, wr, wr, C)
              .permute(0, 3, 1, 2))

        x = self.attn(x)

        x = (x.permute(0, 2, 3, 1)
              .view(B, nH, nW, wr, wr, C)
              .transpose(2, 3)
              .reshape(B, pH, pW, C))

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W].contiguous()

        return x.permute(0, 3, 1, 2)


class PSABlock_SDFAPB(nn.Module):
    def __init__(self, c, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = LocalWindowAttention_SDFAPB(c, num_heads=num_heads)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x)  if self.add else self.ffn(x)
        return x


class C2PSA_SDFAPB(nn.Module):
    """
    C2PSA with Scale-Decoupled Frequency-Aware Position Bias
    直接替换 YOLO11 backbone 末端的 C2PSA
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(
            PSABlock_SDFAPB(self.c, num_heads=max(self.c // 64, 1))
            for _ in range(n)
        ))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
