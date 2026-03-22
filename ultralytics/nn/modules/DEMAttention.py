import torch
from torch import nn
 
 
class DEMAttention(nn.Module):
    def __init__(self, channels, factor=32):
        super(DEMAttention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
 
        # 多尺度卷积核
        self.dconv5_5 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=5, padding=2,
                                  groups=channels // self.groups)
        self.dconv1_7 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=(1, 7), padding=(0, 3),
                                  groups=channels // self.groups)
        self.dconv7_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=(7, 1), padding=(3, 0),
                                  groups=channels // self.groups)
        self.dconv1_11 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=(1, 11),
                                   padding=(0, 5), groups=channels // self.groups)
        self.dconv11_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=(11, 1),
                                   padding=(5, 0), groups=channels // self.groups)
 
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
 
        # 行、列池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
 
        # 多尺度卷积操作
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
 
        # 多尺度卷积特征提取
        x_init = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        x1 = self.dconv5_5(x_init)
        x2 = self.dconv1_7(x_init)
        x3 = self.dconv7_1(x_init)
        x4 = self.dconv1_11(x_init)
        x5 = self.dconv11_1(x_init)
 
        # 将多尺度卷积结果相加
        x_multi_scale = x1 + x2 + x3 + x4 + x5 + x_init
 
        # 使用GN正则化后的卷积和Softmax操作
        x_multi_scale_gn = self.gn(x_multi_scale)
        x2_conv = self.conv3x3(x_multi_scale_gn)
 
        # 计算权重
        x11 = self.softmax(self.agp(x_multi_scale_gn).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2_conv.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2_conv).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x_multi_scale_gn.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
 
        # 输出特征
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
 
 
 
