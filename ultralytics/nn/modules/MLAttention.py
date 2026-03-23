import torch
import torch.nn as nn
from torch.nn import functional as F
 
class MLAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
        # Linear Attention
        self.phi_q = nn.Linear(dim, dim)
        self.phi_k = nn.Linear(dim, dim)
        self.phi_v = nn.Linear(dim, dim)
 
    def forward(self, x):
        u = x.clone()
 
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
 
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
 
        # Linear Attention
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
 
        Q = self.phi_q(x_flat)  # (B, N, C)
        K = self.phi_k(x_flat)  # (B, N, C)
        V = self.phi_v(x_flat)  # (B, N, C)
 
        # 线性注意力：通过Softmax计算权重
        K_T = K.permute(0, 2, 1)  # (B, C, N)
        attn_weights = torch.matmul(Q, K_T)  # (B, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
 
        attn_output = torch.matmul(attn_weights, V)  # (B, N, C)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # reshape回原来形状
 
        return attn_output * attn + u
 
    
