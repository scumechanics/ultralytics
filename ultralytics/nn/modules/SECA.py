import torch
from torch import nn
from torch.nn.parameter import Parameter
 
 
class SECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(SECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
        # Initialize the layers for Softmax Attention
        self.q_linear = nn.Linear(channel, channel)
        self.k_linear = nn.Linear(channel, channel)
        self.v_linear = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x):
        # ECA attention
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        eca_out = x * y.expand_as(x)
 
        # Softmax Attention
        batch_size, channels, height, width = eca_out.size()
        q = self.q_linear(eca_out.view(batch_size, channels, -1).transpose(1, 2))  # N x (H*W) x C
        k = self.k_linear(eca_out.view(batch_size, channels, -1).transpose(1, 2))  # N x (H*W) x C
        v = self.v_linear(eca_out.view(batch_size, channels, -1).transpose(1, 2))  # N x (H*W) x C
 
        attn = torch.matmul(q, k.transpose(-2, -1))  # N x (H*W) x (H*W)
        attn = self.softmax(attn)
        attn_out = torch.matmul(attn, v)  # N x (H*W) x C
        attn_out = attn_out.transpose(1, 2).view(batch_size, channels, height, width)
 
        return attn_out
 
