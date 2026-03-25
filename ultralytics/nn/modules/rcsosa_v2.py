import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["C3k2_RepVGG", "RCSOSA_V2", "ECA"]

# -----------------------------
# Basic utils
# -----------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -----------------------------
# ECA Attention
# -----------------------------
class ECA(nn.Module):
    """
    Efficient Channel Attention
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                                  # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)                  # [B,1,C]
        y = self.conv1d(y)                                    # [B,1,C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        return x * y.expand_as(x)

# -----------------------------
# RepVGG block
# -----------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(out_channels))
    return result

class RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.SiLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                         dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if (out_channels == in_channels and stride == 1) else None
            self.rbr_dense = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = conv_bn(in_channels, out_channels, 1, stride, padding_11, groups)

    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(x))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    # ---- optional fuse helpers ----
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid
        return kernel, bias

# -----------------------------
# SR: Shuffle-Rep block
# -----------------------------
class SR(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 % 2 == 0 and c2 % 2 == 0, "SR requires even channels."
        self.repconv = RepVGG(c1 // 2, c2 // 2)

    @staticmethod
    def channel_shuffle(x, groups=2):
        b, c, h, w = x.size()
        assert c % groups == 0
        x = x.view(b, groups, c // groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        out = self.channel_shuffle(out, 2)
        return out

# -----------------------------
# RCSOSA-V2 (with ECA)
# -----------------------------
class RCSOSA_V2(nn.Module):
    """
    V2 improvements:
    1) stable n_ = max(1, n//2)
    2) optional ECA attention after aggregation
    3) RepVGG input/output projection
    """
    def __init__(self, c1, c2, n=2, e=0.5, use_eca=True):
        super().__init__()
        n_ = max(1, n // 2)
        c_ = make_divisible(int(c1 * e), 8)

        self.conv1 = RepVGG(c1, c_)                 # proj in
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.conv3 = RepVGG(c_ * 3, c2)             # aggregate out
        self.eca = ECA(c2) if use_eca else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        y = torch.cat((x1, x2, x3), dim=1)
        y = self.conv3(y)
        y = self.eca(y)
        return y

# -----------------------------
# C3k2_RepVGG
# -----------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RepVGG(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2_RepVGG(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 简化版：直接复用 Bottleneck，和你原始风格一致
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


if __name__ == "__main__":
    x = torch.randn(1, 128, 80, 80)
    m = RCSOSA_V2(128, 128, n=2, e=0.5, use_eca=True)
    y = m(x)
    print(y.shape)  # [1,128,80,80]
