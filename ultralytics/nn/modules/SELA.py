import torch
import torch.nn as nn
 
 
class SELA(nn.Module):
    def __init__(self, channels):
        super(SELA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = channels ** -0.5
 
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        ela_out = x * x_h * x_w
 
        # Softmax Attention
        qkv = self.qkv(x).reshape(b, 3 * c, h * w)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
 
        attn = torch.bmm(q, k) * self.scale
        attn = self.softmax(attn)
        softmax_out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
 
        out = ela_out + softmax_out
 
        return out
