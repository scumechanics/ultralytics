import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCoordAtt(nn.Module):
    """高效双坐标注意力模块"""
    def __init__(self, inp, oup, groups=32, reduction=4):
        super(EfficientCoordAtt, self).__init__()
        # 统一池化操作
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // groups)
        
        # 共享卷积减少参数
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)  # 比ReLU更平滑
        
        # 分别处理H和W
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(inp // reduction, inp, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 空间注意力分支（共享前置卷积）
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 拼接后一次性处理
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 分离并生成注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # 空间注意力（避免expand）
        spatial_att = a_h * a_w
        
        # 通道注意力
        channel_att = self.channel_att(x)
        
        # 自适应融合
        out = identity + self.alpha * (identity * spatial_att) + self.beta * (identity * channel_att)
        
        return out


class DCAFE_V2(nn.Module):
    """优化版DCAFE - 空间+通道双重注意力"""
    def __init__(self, in_channels, reduction=4):
        super(DCAFE_V2, self).__init__()
        self.coord_att = EfficientCoordAtt(in_channels, in_channels, reduction=reduction)

    def forward(self, x):
        return self.coord_att(x)

class LightweightCoordAtt(nn.Module):
    """轻量化坐标注意力（适合移动端）"""
    def __init__(self, inp, groups=32):
        super(LightweightCoordAtt, self).__init__()
        mip = max(8, inp // groups)
        
        # 使用深度可分离卷积
        self.dw_conv = nn.Conv2d(inp, inp, kernel_size=3, padding=1, groups=inp)
        self.pw_conv = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        
        # 轻量化注意力生成
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)
        
        # 使用全局池化代替AdaptivePool
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 深度可分离提取特征
        feat = self.act(self.bn(self.pw_conv(self.dw_conv(x))))
        
        # 全局信息
        global_feat = self.global_pool(feat)
        
        # H方向注意力
        h_feat = F.adaptive_avg_pool2d(feat, (h, 1))
        a_h = self.conv_h(h_feat).sigmoid()
        
        # W方向注意力
        w_feat = F.adaptive_avg_pool2d(feat, (1, w))
        a_w = self.conv_w(w_feat).sigmoid()
        
        # 融合
        out = identity * a_h * a_w
        
        return identity + out


class DCAFE_Lite(nn.Module):
    """轻量化DCAFE"""
    def __init__(self, in_channels):
        super(DCAFE_Lite, self).__init__()
        self.coord_att = LightweightCoordAtt(in_channels)

    def forward(self, x):
        return self.coord_att(x)
class HybridCoordAtt(nn.Module):
    """混合注意力机制 - Mean/Max/全局池化三路融合"""
    def __init__(self, inp, oup, groups=32):
        super(HybridCoordAtt, self).__init__()
        mip = max(8, inp // groups)
        
        # 三种池化策略
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(inp, mip, 1, bias=False),
            nn.BatchNorm2d(mip),
            nn.SiLU(inplace=True)
        )
        
        # 多路解码器
        self.decoder_h_mean = nn.Conv2d(mip, oup, 1)
        self.decoder_w_mean = nn.Conv2d(mip, oup, 1)
        self.decoder_h_max = nn.Conv2d(mip, oup, 1)
        self.decoder_w_max = nn.Conv2d(mip, oup, 1)
        self.decoder_global = nn.Conv2d(mip, oup, 1)
        
        # 自适应融合模块
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(oup * 3, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Mean路径
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = self.encoder(torch.cat([x_h_mean, x_w_mean], dim=2))
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        att_mean = self.decoder_h_mean(x_h_mean).sigmoid() * self.decoder_w_mean(x_w_mean).sigmoid()
        
        # Max路径
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = self.encoder(torch.cat([x_h_max, x_w_max], dim=2))
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        att_max = self.decoder_h_max(x_h_max).sigmoid() * self.decoder_w_max(x_w_max).sigmoid()
        
        # 全局路径
        global_feat = self.global_pool(x)
        global_feat = self.encoder(global_feat)
        att_global = self.decoder_global(global_feat).sigmoid()
        
        # 自适应融合三路注意力
        combined_att = torch.cat([att_mean, att_max, att_global], dim=1)
        final_att = self.fusion_conv(combined_att)
        
        return identity + identity * final_att


class DCAFE_Hybrid(nn.Module):
    """混合注意力DCAFE"""
    def __init__(self, in_channels):
        super(DCAFE_Hybrid, self).__init__()
        self.coord_att = HybridCoordAtt(in_channels, in_channels)

    def forward(self, x):
        return self.coord_att(x)
