import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientLinearAttention(nn.Module):
    """真正的线性复杂度注意力"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 使用kernel特征映射降低复杂度
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU()
        )
    
    def forward(self, q, k, v):
        B, C, H, W = q.shape
        N = H * W
        
        # Reshape for multi-head
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        
        # 应用kernel特征映射 φ(q), φ(k)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # 线性注意力: φ(q) @ (φ(k)^T @ v) - O(n)复杂度
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, heads, head_dim, head_dim)
        out = torch.matmul(q, kv)  # (B, heads, N, head_dim)
        
        # Reshape back
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        return out


class AdaptiveSpatialAttention(nn.Module):
    """多尺度空间注意力，适应不同尺寸的病害"""
    def __init__(self, channels, rate=4):
        super().__init__()
        mid_channels = channels // rate
        
        # 多尺度卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        concat = torch.cat([b1, b2, b3], dim=1)
        att = self.fusion(concat)
        return att


class ImprovedLGAM(nn.Module):
    """改进版LGAM，针对道路病害检测优化"""
    def __init__(self, in_channels, rate=4, num_heads=4):
        super().__init__()
        out_channels = in_channels
        mid_channels = in_channels // rate
        
        # 通道注意力 - 使用真正的线性注意力
        self.channel_proj_q = nn.Conv2d(in_channels, in_channels, 1)
        self.channel_proj_k = nn.Conv2d(in_channels, in_channels, 1)
        self.channel_proj_v = nn.Conv2d(in_channels, in_channels, 1)
        self.channel_attention = EfficientLinearAttention(in_channels, num_heads)
        
        # 通道特征增强 - 减少降维损失
        self.channel_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 多尺度自适应
        self.spatial_attention = AdaptiveSpatialAttention(in_channels, rate)
        
        # 残差连接权重
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        identity = x
        
        # 通道注意力分支
        q = self.channel_proj_q(x)
        k = self.channel_proj_k(x)
        v = self.channel_proj_v(x)
        channel_att = self.channel_attention(q, k, v)
        channel_enhance = self.channel_enhance(x)
        x_channel = x * channel_att * channel_enhance
        
        # 空间注意力分支
        spatial_att = self.spatial_attention(x_channel)
        x_spatial = x_channel * spatial_att
        
        # 残差连接
        out = identity + self.alpha * x_channel + self.beta * x_spatial
        
        return out
