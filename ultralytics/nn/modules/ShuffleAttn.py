import torch
import torch.nn as nn
# 序列洗牌注意力模块（SSA）
# https://arxiv.org/pdf/2412.20066
class ShuffleAttn(nn.Module): 
    def __init__(self, in_features, out_features, group=4, input_resolution=(64,64)): 
        super().__init__() 
        self.group = group 
        self.gating = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid() 
        ) 
    
    def channel_shuffle(self, x): 
        batchsize, num_channels, height, width = x.shape 
        assert num_channels % self.group == 0 
        group_channels = num_channels // self.group 
        
        x = x.reshape(batchsize, group_channels, self.group, height, width) 
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
        x = x.reshape(batchsize, num_channels, height, width) 

        return x 
    
    def channel_rearrange(self, x): 
        batchsize, num_channels, height, width = x.shape 
        assert num_channels % self.group == 0 
        group_channels = num_channels // self.group 
        
        x = x.reshape(batchsize, self.group, group_channels, height, width) 
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
        x = x.reshape(batchsize, num_channels, height, width) 

        return x

    def forward(self, x):
        # 1. 计算注意力权重
        attn = self.gating(x)
        # 2. 将注意力权重应用到特征上
        x = x * attn
        # 3. 对通道进行 Shuffle，促进组间特征交流
        x = self.channel_shuffle(x)
        return x

