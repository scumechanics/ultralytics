# defect_focus.py
"""
DefectFocus: 缺陷焦点注意力模块 for YOLO11
===========================================

三大核心创新：
1. 前景-背景解耦注意力 (FG-BG Decoupling Attention)
2. 困难样本挖掘注意力 (Hard-Mining Attention)  
3. 边缘增强模块 (Edge Enhancement Module)

作者: [Your Name]
日期: 2026-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancementModule(nn.Module):
    """
    边缘增强模块
    
    使用Sobel和Laplacian算子提取缺陷边缘特征，
    为注意力机制提供边缘先验信息。
    
    Args:
        None
        
    Returns:
        edge: (B, 1, H, W) 归一化的边缘强度图
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel算子 - 检测水平和垂直边缘
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)
        
        # Laplacian算子 - 检测各方向边缘
        laplacian = torch.tensor([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=torch.float32)
        
        # 注册为buffer，不参与梯度更新
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        
        # 可学习的边缘融合层
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
    
    def rgb_to_gray(self, x):
        """RGB转灰度图"""
        if x.size(1) >= 3:
            # 标准RGB转灰度公式
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x[:, 0:1]
        return gray
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
            
        Returns:
            edge: (B, 1, H, W) 归一化边缘图
        """
        # 转换为灰度图
        gray = self.rgb_to_gray(x)  # (B, 1, H, W)
        
        # Sobel边缘检测
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        sobel_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        # Laplacian边缘检测
        laplacian_edge = F.conv2d(gray, self.laplacian, padding=1).abs()
        
        # 融合两种边缘
        edge_combined = self.edge_fusion(
            torch.cat([sobel_magnitude, laplacian_edge], dim=1)
        )
        
        # 归一化到[0, 1]
        B = edge_combined.shape[0]
        edge_flat = edge_combined.view(B, -1)
        edge_min = edge_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_max = edge_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_normalized = (edge_combined - edge_min) / (edge_max - edge_min + 1e-8)
        
        return edge_normalized


class DefectHeatmapPredictor(nn.Module):
    """
    轻量级缺陷热力图预测分支
    
    通过三层卷积预测每个像素位置存在缺陷的概率，
    作为前景-背景分离的先验。
    
    Args:
        in_channels: 输入通道数
        hidden_dim: 隐藏层通道数
        
    Returns:
        heatmap: (B, 1, H, W) 缺陷概率热力图，范围[0, 1]
    """
    
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            # 第一层：降维
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # 第二层：特征提取
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # 第三层：预测热力图
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
            
        Returns:
            heatmap: (B, 1, H, W) 缺陷概率热力图
        """
        return self.predictor(x)


class ForegroundBackgroundAttention(nn.Module):
    """
    前景-背景解耦注意力
    
    结合热力图预测和边缘增强，生成空间注意力权重，
    突出缺陷区域（前景），抑制背景干扰。
    
    Args:
        in_channels: 输入通道数
        
    Returns:
        out: 注意力加权后的特征
        attention_map: 注意力权重图
        heatmap: 缺陷热力图
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        # 热力图预测分支
        self.heatmap_branch = DefectHeatmapPredictor(in_channels)
        
        # 边缘增强模块
        self.edge_module = EdgeEnhancementModule()
        
        # 热力图和边缘图融合
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 可学习的残差缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
            
        Returns:
            out: (B, C, H, W) 注意力加权特征
            attention_map: (B, 1, H, W) 注意力权重
            heatmap: (B, 1, H, W) 缺陷热力图
        """
        # 预测缺陷热力图
        heatmap = self.heatmap_branch(x)  # (B, 1, H, W)
        
        # 提取边缘特征
        edge = self.edge_module(x)  # (B, 1, H, W)
        
        # 尺寸对齐（如果需要）
        if edge.shape[-2:] != heatmap.shape[-2:]:
            edge = F.interpolate(
                edge, 
                size=heatmap.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 融合热力图和边缘图生成注意力
        attention_map = self.attention_fusion(
            torch.cat([heatmap, edge], dim=1)
        )  # (B, 1, H, W)
        
        # 应用注意力（残差连接）
        out = x + self.gamma * (x * attention_map)
        
        return out, attention_map, heatmap


class HardMiningAttention(nn.Module):
    """
    困难样本挖掘注意力
    
    动态识别低置信度但高IoU的困难区域，
    在训练时增加这些区域的权重。
    
    Args:
        in_channels: 输入通道数
        num_classes: 缺陷类别数
        
    Returns:
        out: 加权后的特征
        confidence: 全局置信度预测
    """
    
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # 全局置信度预测
        self.global_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_classes),
            nn.Sigmoid()
        )
        
        # 空间不确定性预测
        self.spatial_uncertainty = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
            
        Returns:
            out: (B, C, H, W) 加权特征
            confidence: (B, num_classes) 全局置信度
        """
        # 预测全局置信度
        confidence = self.global_confidence(x)  # (B, num_classes)
        
        # 预测空间不确定性
        uncertainty = self.spatial_uncertainty(x)  # (B, 1, H, W)
        
        if self.training:
            # 训练模式：对低置信度区域加权
            # 全局不确定性：置信度越低，不确定性越高
            global_uncertainty = (1.0 - confidence.mean(dim=1, keepdim=True))  # (B, 1)
            global_uncertainty = global_uncertainty.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            
            # 困难样本权重：结合全局和空间不确定性
            hard_weight = 1.0 + global_uncertainty * uncertainty * 2.0
            
            # 应用权重
            out = x * hard_weight
        else:
            # 推理模式：不加权
            out = x
        
        return out, confidence


class DefectFocusAttention(nn.Module):
    """
    DefectFocus完整注意力模块
    
    整合三大创新点：
    1. 前景-背景解耦注意力
    2. 边缘增强
    3. 困难样本挖掘
    
    Args:
        in_channels: 输入通道数
        num_classes: 缺陷类别数（默认3：蜂窝/麻面/露筋）
        use_heatmap: 是否使用热力图分支
        use_edge: 是否使用边缘增强
        use_hard_mining: 是否使用困难样本挖掘
        
    Returns:
        训练时: (out_features, auxiliary_dict)
        推理时: out_features
    """
    
    def __init__(
        self, 
        in_channels, 
        num_classes=3,
        use_heatmap=True,
        use_edge=True,
        use_hard_mining=True
    ):
        super().__init__()
        
        self.use_heatmap = use_heatmap
        self.use_edge = use_edge
        self.use_hard_mining = use_hard_mining
        
        # 前景-背景解耦注意力（包含热力图和边缘）
        if use_heatmap or use_edge:
            self.fg_bg_attention = ForegroundBackgroundAttention(in_channels)
        
        # 困难样本挖掘注意力
        if use_hard_mining:
            self.hard_mining_attention = HardMiningAttention(in_channels, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
            
        Returns:
            训练时: (out, aux_dict)
            推理时: out
        """
        auxiliary_outputs = {}
        
        # 步骤1: 前景-背景解耦注意力
        if self.use_heatmap or self.use_edge:
            x, attention_map, heatmap = self.fg_bg_attention(x)
            auxiliary_outputs['attention_map'] = attention_map
            auxiliary_outputs['heatmap'] = heatmap
        
        # 步骤2: 困难样本挖掘注意力
        if self.use_hard_mining:
            x, confidence = self.hard_mining_attention(x)
            auxiliary_outputs['confidence'] = confidence
        
        # 训练时返回辅助输出，推理时只返回特征
        if self.training:
            return x, auxiliary_outputs
        else:
            return x


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("DefectFocus 模块测试")
    print("="*60)
    
    # 创建测试输入
    batch_size = 2
    channels = 256
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试完整模块
    print("\n1. 测试完整DefectFocus模块")
    model = DefectFocusAttention(
        in_channels=channels,
        num_classes=3,
        use_heatmap=True,
        use_edge=True,
        use_hard_mining=True
    )
    
    # 训练模式
    model.train()
    out_train, aux = model(x)
    print(f"   训练模式输出: {out_train.shape}")
    print(f"   辅助输出: {list(aux.keys())}")
    if 'attention_map' in aux:
        print(f"   - attention_map: {aux['attention_map'].shape}")
    if 'heatmap' in aux:
        print(f"   - heatmap: {aux['heatmap'].shape}")
    if 'confidence' in aux:
        print(f"   - confidence: {aux['confidence'].shape}")
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"   推理模式输出: {out_eval.shape}")
    
    # 测试各个子模块
    print("\n2. 测试边缘增强模块")
    edge_module = EdgeEnhancementModule()
    edge = edge_module(x)
    print(f"   边缘图: {edge.shape}, 范围: [{edge.min():.3f}, {edge.max():.3f}]")
    
    print("\n3. 测试热力图预测")
    heatmap_module = DefectHeatmapPredictor(channels)
    heatmap = heatmap_module(x)
    print(f"   热力图: {heatmap.shape}, 范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print("\n4. 测试前景-背景注意力")
    fg_bg = ForegroundBackgroundAttention(channels)
    out_fg, attn, heat = fg_bg(x)
    print(f"   输出特征: {out_fg.shape}")
    print(f"   注意力图: {attn.shape}")
    print(f"   热力图: {heat.shape}")
    
    print("\n5. 测试困难样本挖掘")
    hard_mining = HardMiningAttention(channels, num_classes=3)
    hard_mining.train()
    out_hard, conf = hard_mining(x)
    print(f"   输出特征: {out_hard.shape}")
    print(f"   置信度: {conf.shape}")
    
    # 参数统计
    print("\n6. 模型参数统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

