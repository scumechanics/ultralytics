import torch
from torch import nn
 
 
# ECA模块
class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
 
 
# PolarizedSelfAttention模块
class PolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                                                 1).reshape(
            b, c, 1, 1)
        channel_out = channel_weight * x
 
        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)
        spatial_wq = self.sp_wq(x)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x
 
        out = spatial_out + channel_out
        return out
 
 
# EPSA模块
class EPSA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(EPSA, self).__init__()
        self.eca = ECA(channel, k_size)
        self.psa = PolarizedSelfAttention(channel)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的权重参数
 
    def forward(self, x):
        eca_out = self.eca(x)
        psa_out = self.psa(x)
        out = self.alpha * eca_out + (1 - self.alpha) * psa_out
        return out
