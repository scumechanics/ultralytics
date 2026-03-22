import torch
import torch.nn as nn
 
 
class DSAttention(nn.Module):
    def __init__(self, channels):
        super(DSAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )
 
        # 多尺度卷积操作
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
 
        # Softmax Attention
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = channels ** -0.5
 
        # 最终融合卷积
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        ela_out = x * x_h * x_w
 
        # 多尺度卷积处理
        x_init = self.dconv5_5(ela_out)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        multi_scale_out = x_1 + x_2 + x_3 + x_init
 
        # Softmax Attention
        qkv = self.qkv(multi_scale_out).reshape(b, 3 * c, h * w)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
 
        attn = torch.bmm(q, k) * self.scale
        attn = self.softmax(attn)
        softmax_out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
 
        out = multi_scale_out + softmax_out
 
        # 最终卷积输出
        out = self.conv(out)
 
        return out
 
