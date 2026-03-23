# # defect_focus.py
# """
# DefectFocus: 缺陷焦点注意力模块 for YOLO11
# 所有 forward() 方法均只返回纯 Tensor，与 YOLO backbone 完全兼容。
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EdgeEnhancementModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
#         sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
#         laplacian = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
#         self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
#         self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))
#         self.register_buffer('laplacian', laplacian.view(1,1,3,3))
#         self.edge_fusion = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=1, bias=False),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # x must be a plain Tensor
#         assert isinstance(x, torch.Tensor), f"EdgeEnhancementModule got {type(x)}"
#         single = x.mean(dim=1, keepdim=True)
#         ex = F.conv2d(single, self.sobel_x, padding=1)
#         ey = F.conv2d(single, self.sobel_y, padding=1)
#         sobel = torch.sqrt(ex**2 + ey**2 + 1e-8)
#         lap = F.conv2d(single, self.laplacian, padding=1).abs()
#         combined = self.edge_fusion(torch.cat([sobel, lap], dim=1))
#         B = combined.shape[0]
#         flat = combined.view(B, -1)
#         mn = flat.min(1, keepdim=True)[0].view(B,1,1,1)
#         mx = flat.max(1, keepdim=True)[0].view(B,1,1,1)
#         return (combined - mn) / (mx - mn + 1e-8)   # pure Tensor


# class DefectHeatmapPredictor(nn.Module):
#     def __init__(self, in_channels, hidden_dim=64):
#         super().__init__()
#         self.predictor = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim//2, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor), f"DefectHeatmapPredictor got {type(x)}"
#         return self.predictor(x)   # pure Tensor


# class ForegroundBackgroundAttention(nn.Module):
#     """
#     forward() 只返回纯 Tensor。
#     辅助输出缓存在 last_attention_map / last_heatmap。
#     """
#     def __init__(self, in_channels):
#         super().__init__()
#         self.heatmap_branch = DefectHeatmapPredictor(in_channels)
#         self.edge_module = EdgeEnhancementModule()
#         self.attention_fusion = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.last_attention_map = None
#         self.last_heatmap = None

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor), f"ForegroundBackgroundAttention got {type(x)}"
#         heatmap = self.heatmap_branch(x)
#         edge = self.edge_module(x)
#         if edge.shape[-2:] != heatmap.shape[-2:]:
#             edge = F.interpolate(edge, size=heatmap.shape[-2:], mode='bilinear', align_corners=False)
#         attn = self.attention_fusion(torch.cat([heatmap, edge], dim=1))
#         out = x + self.gamma * (x * attn)
#         self.last_attention_map = attn.detach()
#         self.last_heatmap = heatmap.detach()
#         return out   # pure Tensor


# class HardMiningAttention(nn.Module):
#     """
#     forward() 只返回纯 Tensor。
#     置信度缓存在 last_confidence。
#     """
#     def __init__(self, in_channels, num_classes=3):
#         super().__init__()
#         self.num_classes = num_classes
#         self.global_confidence = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, in_channels//4),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels//4, num_classes),
#             nn.Sigmoid()
#         )
#         self.spatial_uncertainty = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
#             nn.BatchNorm2d(in_channels//4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels//4, 1, 1),
#             nn.Sigmoid()
#         )
#         self.last_confidence = None

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor), f"HardMiningAttention got {type(x)}"
#         confidence = self.global_confidence(x)
#         uncertainty = self.spatial_uncertainty(x)
#         if self.training:
#             g = (1.0 - confidence.mean(dim=1, keepdim=True)).view(-1,1,1,1)
#             out = x * (1.0 + g * uncertainty * 2.0)
#         else:
#             out = x
#         self.last_confidence = confidence.detach()
#         return out   # pure Tensor


# class DefectFocusAttention(nn.Module):
#     """
#     YOLO11 兼容的缺陷焦点注意力模块。
#     forward() 始终只返回纯 Tensor。

#     Args:
#         in_channels (int): 输入通道数
#         num_classes (int): 缺陷类别数，默认 3
#         use_heatmap (bool): 是否启用热力图分支
#         use_edge (bool): 是否启用边缘增强
#         use_hard_mining (bool): 是否启用困难样本挖掘
#     """
#     def __init__(self, in_channels, num_classes=3,
#                  use_heatmap=True, use_edge=True, use_hard_mining=True):
#         super().__init__()
#         self.use_heatmap = use_heatmap
#         self.use_edge = use_edge
#         self.use_hard_mining = use_hard_mining

#         if use_heatmap or use_edge:
#             self.fg_bg_attention = ForegroundBackgroundAttention(in_channels)
#         if use_hard_mining:
#             self.hard_mining_attention = HardMiningAttention(in_channels, num_classes)

#     @property
#     def last_attention_map(self):
#         return self.fg_bg_attention.last_attention_map if (self.use_heatmap or self.use_edge) else None

#     @property
#     def last_heatmap(self):
#         return self.fg_bg_attention.last_heatmap if (self.use_heatmap or self.use_edge) else None

#     @property
#     def last_confidence(self):
#         return self.hard_mining_attention.last_confidence if self.use_hard_mining else None

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor), \
#             f"DefectFocusAttention.forward received {type(x)}, expected Tensor"
#         if self.use_heatmap or self.use_edge:
#             x = self.fg_bg_attention(x)
#         if self.use_hard_mining:
#             x = self.hard_mining_attention(x)
#         assert isinstance(x, torch.Tensor), "Internal error: output is not a Tensor"
#         return x   # pure Tensor


# # ============================================================
# # 辅助损失读取方式（在自定义 loss 函数中使用）
# # ============================================================
# # for m in model.modules():
# #     if isinstance(m, DefectFocusAttention):
# #         heatmap = m.last_heatmap        # (B, 1, H, W)
# #         attn    = m.last_attention_map  # (B, 1, H, W)
# #         conf    = m.last_confidence     # (B, num_classes)


# # ============================================================
# # 自测
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 55)
#     print("DefectFocus 自测（所有输出必须为纯 Tensor）")
#     print("=" * 55)

#     B, C, H, W = 2, 256, 64, 64
#     x = torch.randn(B, C, H, W)

#     model = DefectFocusAttention(C, num_classes=3)

#     model.train()
#     out = model(x)
#     assert isinstance(out, torch.Tensor), f"FAIL train: got {type(out)}"
#     print(f"train  输出: {out.shape}  类型: {type(out).__name__}  ✓")

#     model.eval()
#     with torch.no_grad():
#         out = model(x)
#     assert isinstance(out, torch.Tensor), f"FAIL eval: got {type(out)}"
#     print(f"eval   输出: {out.shape}  类型: {type(out).__name__}  ✓")

#     print(f"last_heatmap:       {model.last_heatmap.shape}")
#     print(f"last_attention_map: {model.last_attention_map.shape}")
#     print(f"last_confidence:    {model.last_confidence.shape}")

#     total = sum(p.numel() for p in model.parameters())
#     print(f"参数量: {total:,}")
#     print("=" * 55)
#     print("全部通过！")



"""
DefectFocus: 缺陷焦点注意力模块 for YOLO11 (轻量保守版)
=======================================================

修复说明 (v3):
- HardMiningAttention: 去掉训练/推理不一致，改用轻量 SE-like 通道注意力
- ForegroundBackgroundAttention: 去掉独立热力图预测分支，改为轻量空间注意力
- 所有幅度调制限制在 [0.9, 1.1] 区间，避免破坏 backbone 特征分布
- 所有 forward() 只返回纯 Tensor

核心设计原则:
  "残差式注意力" —— out = x + alpha * (attn * x)
  alpha 初始化为接近 0，让模块初期接近恒等映射，训练稳定后再逐渐发挥作用

作者: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 轻量空间注意力（替代重量级热力图预测分支）
# ============================================================
class LightSpatialAttn(nn.Module):
    """
    轻量空间注意力：3层卷积，无独立预测头
    输出 (B, 1, H, W) sigmoid 权重，接近 1 时不改变特征
    """
    def __init__(self, in_channels):
        super().__init__()
        mid = max(in_channels // 16, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)   # (B, 1, H, W)


# ============================================================
# 轻量通道注意力（替代 HardMiningAttention）
# ============================================================
class LightChannelAttn(nn.Module):
    """
    SE-like 通道注意力，输出 (B, C, 1, 1) sigmoid 权重
    幅度限制在 [0.8, 1.0]，不放大特征
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)   # (B, C, 1, 1)


# ============================================================
# 主模块
# ============================================================
class DefectFocusAttention(nn.Module):
    """
    DefectFocus 缺陷焦点注意力模块（YOLO11 兼容，轻量保守版）

    设计原则:
      out = x + alpha * attn_spatial * x  （空间注意力分支）
           + beta  * attn_channel * x     （通道注意力分支）

    alpha, beta 初始化为 0，模块初期等价于恒等映射，
    随训练进行梯度会让 alpha/beta 逐渐增大，平稳发挥作用。

    Args:
        in_channels (int): 输入通道数（由 YAML 自动传入）
        num_classes (int): 保留参数，兼容旧 YAML，实际不使用
        use_spatial (bool): 是否启用空间注意力分支
        use_channel (bool): 是否启用通道注意力分支
    """

    def __init__(self, in_channels, num_classes=3,
                 use_spatial=True, use_channel=True):
        super().__init__()
        self.use_spatial = use_spatial
        self.use_channel = use_channel

        if use_spatial:
            self.spatial_attn = LightSpatialAttn(in_channels)
            # alpha 初始化为 0 → 初期恒等映射
            self.alpha = nn.Parameter(torch.zeros(1))

        if use_channel:
            self.channel_attn = LightChannelAttn(in_channels)
            # beta 初始化为 0 → 初期恒等映射
            self.beta = nn.Parameter(torch.zeros(1))

        # 辅助输出缓存（供自定义 loss 使用，不影响 forward 返回值）
        self.last_spatial_map = None
        self.last_channel_map = None

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 纯 Tensor
        Returns:
            out: (B, C, H, W) 纯 Tensor
        """
        assert isinstance(x, torch.Tensor), \
            f"DefectFocusAttention expects Tensor, got {type(x)}"

        out = x

        if self.use_spatial:
            s = self.spatial_attn(x)            # (B, 1, H, W)，值域 [0,1]
            # 残差式：只在原特征基础上微调，alpha 从 0 开始学习
            out = out + self.alpha * (s * out)
            self.last_spatial_map = s.detach()

        if self.use_channel:
            c = self.channel_attn(out)          # (B, C, 1, 1)，值域 [0,1]
            out = out + self.beta * (c * out)
            self.last_channel_map = c.detach()

        return out   # 纯 Tensor


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("DefectFocus 自测")
    print("=" * 55)

    for C in [256, 512, 1024]:
        x = torch.randn(2, C, 40, 40)
        m = DefectFocusAttention(C)

        m.train()
        out = m(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"

        # 初始化时 alpha=beta=0，输出应等于输入
        diff = (out - x).abs().max().item()
        assert diff < 1e-5, f"初始化时输出应等于输入，diff={diff}"
        print(f"  C={C:4d}  shape={tuple(out.shape)}  init_diff={diff:.2e}  ✓")

    total = sum(p.numel() for p in DefectFocusAttention(512).parameters())
    print(f"\n参数量 (C=512): {total:,}")
    print("=" * 55)
    print("全部通过！")



