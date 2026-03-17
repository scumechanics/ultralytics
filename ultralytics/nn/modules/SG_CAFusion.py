# ultralytics/nn/Addmodules/SG_CAFusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SG_CAFusion', 'SaliencyAuxLoss']


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True
                    else act if isinstance(act, nn.Module)
                    else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ─── 语义显著性估计 ──────────────────────────────────────────

class SemanticSaliencyEstimator(nn.Module):
    """
    从高层特征提取前景概率显著图

    设计：轻量两层conv，输出 (B,1,H,W) 的0-1显著图
    训练时可用GT bbox生成的粗糙mask做辅助监督
    推理时显著图同时用于调制融合权重
    """

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(),
            nn.Conv2d(c // 2, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # (B, 1, H, W)


# ─── 语义引导融合门控 ────────────────────────────────────────

class SemanticGuidedFusionGate(nn.Module):
    """
    语义引导内容自适应融合门控

    与普通动态门控的核心差异：
      普通版：gate输入 = concat(f_high, f_low)，纯外观驱动
      本版本：gate输入 = concat(f_high, f_low, saliency)
              语义显著图作为先验，让前景区域更信任高层语义，
              背景区域更信任低层细节

    互补约束：Softmax保证 w_high + w_low = 1（逐像素）
    """

    def __init__(self, c):
        super().__init__()
        # 输入：2c（两路特征）+ 1（显著图）
        self.gate = nn.Sequential(
            nn.Conv2d(c * 2 + 1, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, 2, 1, bias=False),
            nn.Softmax(dim=1)   # 互补约束
        )

    def forward(self, f_high, f_low, saliency):
        gate_in = torch.cat([f_high, f_low, saliency], dim=1)
        weights = self.gate(gate_in)          # (B, 2, H, W)
        w_high  = weights[:, 0:1]
        w_low   = weights[:, 1:2]
        return f_high * w_high + f_low * w_low


# ─── 主模块 ─────────────────────────────────────────────────

class SG_CAFusion(nn.Module):
    """
    Semantic-Guided Content-Adaptive Cross-Scale Fusion

    替换 FPN/PAN 中的 Upsample + Concat + Conv

    流程：
      1. 空间对齐（上采样或下采样）
      2. 通道投影到统一维度
      3. 语义显著性估计（从高层特征）
      4. 语义引导动态门控融合
      5. 输出精炼 + 残差连接

    Args:
        c_high   (int): 高层特征通道数
        c_low    (int): 低层特征通道数
        c_out    (int): 输出通道数
        mode     (str): 'up' FPN方向 / 'down' PAN方向
        aux_loss (bool): True时forward返回(out, saliency)用于辅助监督
    """

    def __init__(self, c_high, c_low, c_out, mode='up', aux_loss=False):
        super().__init__()
        self.mode     = mode
        self.aux_loss = aux_loss

        self.proj_high = Conv(c_high, c_out, 1)
        self.proj_low  = Conv(c_low,  c_out, 1)

        # 语义显著性估计（核心组件1）
        self.saliency_est = SemanticSaliencyEstimator(c_out)

        # 语义引导融合门控（核心组件2）
        self.gate = SemanticGuidedFusionGate(c_out)

        # 输出精炼
        self.refine = Conv(c_out, c_out, 3)

        # 残差对齐
        c_in = c_high + c_low
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            if c_in != c_out else nn.Identity()
        )

    def forward(self, inputs):
        """
        inputs: [x_high, x_low]
          x_high: 高层语义特征 (B, c_high, H_h, W_h)
          x_low:  低层细节特征 (B, c_low,  H_l, W_l)

        returns:
          aux_loss=False → out (B, c_out, H, W)
          aux_loss=True  → (out, saliency) 用于辅助loss计算
        """
        x_high, x_low = inputs

        # step1: 空间对齐
        if self.mode == 'up':
            x_high_a = F.interpolate(x_high, size=x_low.shape[2:], mode='nearest')
            x_low_a  = x_low
        else:
            x_high_a = x_high
            x_low_a  = F.adaptive_avg_pool2d(x_low, x_high.shape[2:])

        # step2: 通道投影
        f_high = self.proj_high(x_high_a)
        f_low  = self.proj_low(x_low_a)

        # step3: 语义显著图
        saliency = self.saliency_est(f_high)   # (B, 1, H, W)

        # step4: 语义引导融合
        fused = self.gate(f_high, f_low, saliency)
        fused = self.refine(fused)

        # step5: 残差
        residual = self.shortcut(
            torch.cat([x_high_a, x_low_a], dim=1)
        )
        out = fused + residual

        if self.aux_loss:
            return out, saliency
        return out


# ─── 辅助监督Loss ────────────────────────────────────────────

class SaliencyAuxLoss(nn.Module):
    """
    显著图辅助监督Loss

    用GT bbox生成粗糙前景mask监督显著图，
    无需额外标注，bbox内为1，外为0。

    Args:
        weight (float): 辅助loss权重，建议0.05~0.1

    Usage:
        aux_loss_fn = SaliencyAuxLoss(weight=0.1)

        # 在训练loop中收集各层显著图
        out_p3, sal_p3 = cafusion_p3([p5, p4], aux_loss=True)
        out_p4, sal_p4 = cafusion_p4([p5, p3], aux_loss=True)

        loss_aux = aux_loss_fn(
            saliency_maps=[sal_p3, sal_p4],
            targets=batch['bboxes'],   # (N, 6): img_idx,cls,cx,cy,w,h
        )
        total_loss = det_loss + loss_aux
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.bce    = nn.BCELoss()

    def _make_mask(self, saliency, targets):
        B, _, H, W = saliency.shape
        mask = torch.zeros(B, 1, H, W, device=saliency.device)

        if targets is None or len(targets) == 0:
            return mask

        for t in targets:
            b  = int(t[0])
            cx, cy, w, h = t[2].item(), t[3].item(), t[4].item(), t[5].item()
            x1 = max(0, int((cx - w / 2) * W))
            y1 = max(0, int((cy - h / 2) * H))
            x2 = min(W, int((cx + w / 2) * W))
            y2 = min(H, int((cy + h / 2) * H))
            if x2 > x1 and y2 > y1:
                mask[b, 0, y1:y2, x1:x2] = 1.0

        return mask

    def forward(self, saliency_maps, targets):
        """
        saliency_maps: list of (B, 1, H, W)
        targets:       tensor (N, 6) YOLO格式
        """
        total = sum(
            self.bce(sal, self._make_mask(sal, targets))
            for sal in saliency_maps
        )
        return self.weight * total / len(saliency_maps)
