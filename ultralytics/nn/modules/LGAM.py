import torch
import torch.nn as nn
 
 
class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super(LinearAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, q, k, v):
        k = k.transpose(-2, -1)
        scores = torch.matmul(q, k)
        scores = self.softmax(scores)
        context = torch.matmul(scores, v)
        return context
 
 
class LGAM(nn.Module):
    def __init__(self, in_channels, rate=4):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)
 
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)
 
        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')
 
        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
 
        self.channel_attention = LinearAttention(d_model=in_channels)
 
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
 
        # Integrate linear attention into channel attention
        x_channel_att = self.channel_attention(x_channel_att, x_channel_att, x_channel_att)
 
        x = x * x_channel_att
 
        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))
 
        out = x * x_spatial_att
 
        return out
 
