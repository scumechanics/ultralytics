import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return identity + attention_mean + attention_max

class DCAFE(nn.Module):
    def __init__(self, in_channels):
        super(DCAFE, self).__init__()
        self.coord_att = CoordAttMeanMax(in_channels, oup=in_channels)

    def forward(self, x):
        return self.coord_att(x)



if __name__ == "__main__":
    # 定义输入张量大小（Batch、Channel、Height、Wight）
    B, C, H, W = 16, 64, 40, 40
    input_tensor = torch.randn(B,C,H,W)  # 随机生成输入张量
    dim=C
    # 创建 ARConv 实例
    block = DCAFE(in_channels=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sablock = block.to(device)
    print(sablock)
    input_tensor = input_tensor.to(device)
    # 执行前向传播
    output = sablock(input_tensor)
    # 打印输入和输出的形状
    print(f"Input: {input_tensor.shape}")
    print(f"Output: {output.shape}")

