import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(d_model // num_heads, 1)
        self.scale = self.head_dim ** -0.5  # 关键：缩放因子

        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ELU()  # ELU比ReLU更稳定，避免全0导致除零
        )
        self.norm = nn.LayerNorm(d_model)  # 输出归一化

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        N = H * W

        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # kernel特征映射 + 数值稳定
        q = self.feature_map(q) + 1e-6
        k = self.feature_map(k) + 1e-6

        # 归一化线性注意力
        kv = torch.matmul(k.transpose(-2, -1), v) * self.scale
        # 归一化因子，防止爆炸
        normalizer = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6
        out = torch.matmul(q, kv) / normalizer

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        # LayerNorm需要在最后一维
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


class AdaptiveSpatialAttention(nn.Module):
    def __init__(self, channels, rate=4):
        super().__init__()
        mid_channels = max(channels // rate, 1)

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(concat)


class ImprovedLGAM(nn.Module):
    def __init__(self, in_channels, rate=4, num_heads=4):
        super().__init__()
        in_channels = int(in_channels)
        # num_heads必须能整除in_channels
        while in_channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.channel_proj_q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.channel_proj_k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.channel_proj_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.channel_attention = EfficientLinearAttention(in_channels, num_heads)

        self.channel_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // rate, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // rate, 1), in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = AdaptiveSpatialAttention(in_channels, rate)

        # 用小初始值，训练初期接近恒等映射
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.out_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x

        q = self.channel_proj_q(x)
        k = self.channel_proj_k(x)
        v = self.channel_proj_v(x)
        channel_att = self.channel_attention(q, k, v)
        channel_enhance = self.channel_enhance(x)
        x_channel = x * channel_att * channel_enhance

        spatial_att = self.spatial_attention(x_channel)
        x_spatial = x_channel * spatial_att

        # alpha/beta初始为0，训练稳定后逐渐增大
        out = identity + self.alpha * x_channel + self.beta * x_spatial
        out = self.out_norm(out)

        return out
