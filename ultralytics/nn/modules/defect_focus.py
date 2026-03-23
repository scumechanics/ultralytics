# defect_focus.py
"""
DefectFocus: 缺陷焦点注意力模块 for YOLO11
所有 forward() 方法均只返回纯 Tensor，与 YOLO backbone 完全兼容。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancementModule(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        laplacian = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))
        self.register_buffer('laplacian', laplacian.view(1,1,3,3))
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x must be a plain Tensor
        assert isinstance(x, torch.Tensor), f"EdgeEnhancementModule got {type(x)}"
        single = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(single, self.sobel_x, padding=1)
        ey = F.conv2d(single, self.sobel_y, padding=1)
        sobel = torch.sqrt(ex**2 + ey**2 + 1e-8)
        lap = F.conv2d(single, self.laplacian, padding=1).abs()
        combined = self.edge_fusion(torch.cat([sobel, lap], dim=1))
        B = combined.shape[0]
        flat = combined.view(B, -1)
        mn = flat.min(1, keepdim=True)[0].view(B,1,1,1)
        mx = flat.max(1, keepdim=True)[0].view(B,1,1,1)
        return (combined - mn) / (mx - mn + 1e-8)   # pure Tensor


class DefectHeatmapPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"DefectHeatmapPredictor got {type(x)}"
        return self.predictor(x)   # pure Tensor


class ForegroundBackgroundAttention(nn.Module):
    """
    forward() 只返回纯 Tensor。
    辅助输出缓存在 last_attention_map / last_heatmap。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.heatmap_branch = DefectHeatmapPredictor(in_channels)
        self.edge_module = EdgeEnhancementModule()
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_attention_map = None
        self.last_heatmap = None

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"ForegroundBackgroundAttention got {type(x)}"
        heatmap = self.heatmap_branch(x)
        edge = self.edge_module(x)
        if edge.shape[-2:] != heatmap.shape[-2:]:
            edge = F.interpolate(edge, size=heatmap.shape[-2:], mode='bilinear', align_corners=False)
        attn = self.attention_fusion(torch.cat([heatmap, edge], dim=1))
        out = x + self.gamma * (x * attn)
        self.last_attention_map = attn.detach()
        self.last_heatmap = heatmap.detach()
        return out   # pure Tensor


class HardMiningAttention(nn.Module):
    """
    forward() 只返回纯 Tensor。
    置信度缓存在 last_confidence。
    """
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.global_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels//4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//4, num_classes),
            nn.Sigmoid()
        )
        self.spatial_uncertainty = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, 1, 1),
            nn.Sigmoid()
        )
        self.last_confidence = None

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"HardMiningAttention got {type(x)}"
        confidence = self.global_confidence(x)
        uncertainty = self.spatial_uncertainty(x)
        if self.training:
            g = (1.0 - confidence.mean(dim=1, keepdim=True)).view(-1,1,1,1)
            out = x * (1.0 + g * uncertainty * 2.0)
        else:
            out = x
        self.last_confidence = confidence.detach()
        return out   # pure Tensor


class DefectFocusAttention(nn.Module):
    """
    YOLO11 兼容的缺陷焦点注意力模块。
    forward() 始终只返回纯 Tensor。

    Args:
        in_channels (int): 输入通道数
        num_classes (int): 缺陷类别数，默认 3
        use_heatmap (bool): 是否启用热力图分支
        use_edge (bool): 是否启用边缘增强
        use_hard_mining (bool): 是否启用困难样本挖掘
    """
    def __init__(self, in_channels, num_classes=3,
                 use_heatmap=True, use_edge=True, use_hard_mining=True):
        super().__init__()
        self.use_heatmap = use_heatmap
        self.use_edge = use_edge
        self.use_hard_mining = use_hard_mining

        if use_heatmap or use_edge:
            self.fg_bg_attention = ForegroundBackgroundAttention(in_channels)
        if use_hard_mining:
            self.hard_mining_attention = HardMiningAttention(in_channels, num_classes)

    @property
    def last_attention_map(self):
        return self.fg_bg_attention.last_attention_map if (self.use_heatmap or self.use_edge) else None

    @property
    def last_heatmap(self):
        return self.fg_bg_attention.last_heatmap if (self.use_heatmap or self.use_edge) else None

    @property
    def last_confidence(self):
        return self.hard_mining_attention.last_confidence if self.use_hard_mining else None

    def forward(self, x):
        assert isinstance(x, torch.Tensor), \
            f"DefectFocusAttention.forward received {type(x)}, expected Tensor"
        if self.use_heatmap or self.use_edge:
            x = self.fg_bg_attention(x)
        if self.use_hard_mining:
            x = self.hard_mining_attention(x)
        assert isinstance(x, torch.Tensor), "Internal error: output is not a Tensor"
        return x   # pure Tensor


# ============================================================
# 辅助损失读取方式（在自定义 loss 函数中使用）
# ============================================================
# for m in model.modules():
#     if isinstance(m, DefectFocusAttention):
#         heatmap = m.last_heatmap        # (B, 1, H, W)
#         attn    = m.last_attention_map  # (B, 1, H, W)
#         conf    = m.last_confidence     # (B, num_classes)


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("DefectFocus 自测（所有输出必须为纯 Tensor）")
    print("=" * 55)

    B, C, H, W = 2, 256, 64, 64
    x = torch.randn(B, C, H, W)

    model = DefectFocusAttention(C, num_classes=3)

    model.train()
    out = model(x)
    assert isinstance(out, torch.Tensor), f"FAIL train: got {type(out)}"
    print(f"train  输出: {out.shape}  类型: {type(out).__name__}  ✓")

    model.eval()
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor), f"FAIL eval: got {type(out)}"
    print(f"eval   输出: {out.shape}  类型: {type(out).__name__}  ✓")

    print(f"last_heatmap:       {model.last_heatmap.shape}")
    print(f"last_attention_map: {model.last_attention_map.shape}")
    print(f"last_confidence:    {model.last_confidence.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
    print("=" * 55)
    print("全部通过！")

