import torch
import torch.nn as nn
 
__all__ = ['MSHC']
 
def _to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)
 
 
def channel_shuffle(x, groups: int):
    """
    x: [B, C, H, W]
    """
    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError(f"channel_shuffle: channels ({c}) must be divisible by groups ({groups}).")
 
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x
 
 
class ChannelSpatialAttention(nn.Module):
    """
    PyTorch equivalent of the active Keras ChannelSpatialAttention in your file.
    Effective behavior preserved:
      - Channel attention: avg + max + min + sum over spatial dims, then 1x1 conv + sigmoid
      - Spatial attention: avg + max + min + sum over channel dim, then 7x7 conv + sigmoid
      - Final gate: sigmoid(channel_info + spatial_info)
      - Output: x * attention_map
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=True)
 
    def forward(self, x):
        # Channel holistic information attention
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = x.amax(dim=(2, 3), keepdim=True)
        min_pool = x.amin(dim=(2, 3), keepdim=True)
        sum_pool = x.sum(dim=(2, 3), keepdim=True)
        pooled = avg_pool + max_pool + min_pool + sum_pool
        channel_info = torch.sigmoid(self.channel_conv(pooled))
 
        # Spatial holistic information attention
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.amax(dim=1, keepdim=True)
        min_spatial = x.amin(dim=1, keepdim=True)
        sum_spatial = x.sum(dim=1, keepdim=True)
        spatial_input = avg_spatial + max_spatial + min_spatial + sum_spatial
        spatial_info = torch.sigmoid(self.spatial_conv(spatial_input))
 
        attention_map = torch.sigmoid(channel_info + spatial_info)
        return x * attention_map
 
 
class MSHC(nn.Module):
    """
    PyTorch equivalent of your active multi_kernel_groupwise_conv1.
    Keras logic mapped to PyTorch:
      1) Four parallel branches:
         - 1x1 Conv2d -> out_channels // 4
         - 3x3 DepthwiseConv2d
         - 5x5 DepthwiseConv2d
         - 3x3 Dilated DepthwiseConv2d (dilation=2)
      2) Concatenate
      3) Channel shuffle
      4) 3x3 depthwise conv
      5) 1x1 grouped pointwise conv -> out_channels
      6) 3x3 depthwise downsample
      7) ChannelSpatialAttention
      8) Shortcut: 1x1 grouped PW -> 3x3 depthwise downsample -> 1x1 grouped PW
      9) Residual add + ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 4,
        stride=2,
        use_attention: bool = True,
    ):
        super().__init__()
        stride = _to_2tuple(stride)
 
        if out_channels % 4 != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by 4.")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups}).")
 
        concat_channels = out_channels // 4 + 3 * in_channels
        if concat_channels % groups != 0:
            raise ValueError(
                f"Concatenated channels ({concat_channels}) must be divisible by groups ({groups}) "
                f"for channel shuffle."
            )
 
        self.groups = groups
 
        # Four heterogeneous branches
        self.branch1 = nn.Conv2d(
            in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.branch2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1,
            groups=in_channels, bias=True
        )
        self.branch3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, stride=1, padding=2,
            groups=in_channels, bias=True
        )
        self.branch4 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2,
            groups=in_channels, bias=True
        )
 
        # Fusion path
        self.fuse_dw = nn.Conv2d(
            concat_channels, concat_channels, kernel_size=3, stride=1, padding=1,
            groups=concat_channels, bias=True
        )
        self.fuse_pw = nn.Conv2d(
            concat_channels, out_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=True
        )
        self.down_dw = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            groups=out_channels, bias=True
        )
 
        self.attention = ChannelSpatialAttention(out_channels) if use_attention else nn.Identity()
 
        # Shortcut path
        self.short_pw1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=True
        )
        self.short_dw = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            groups=out_channels, bias=True
        )
        self.short_pw2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=True
        )
 
        self.act = nn.ReLU(inplace=True)
 
    def forward(self, x):
        # Multi-branch extraction
        conv1x1 = self.branch1(x)
        conv3x3 = self.branch2(x)
        conv5x5 = self.branch3(x)
        conv_dilated = self.branch4(x)
 
        # Concatenate + shuffle
        y = torch.cat([conv1x1, conv3x3, conv5x5, conv_dilated], dim=1)
        y = channel_shuffle(y, self.groups)
 
        # Fusion path
        y = self.fuse_dw(y)
        y = self.fuse_pw(y)
        y = self.down_dw(y)
        y = self.attention(y)
 
        # Shortcut path
        s = self.short_pw1(x)
        s = self.short_dw(s)
        s = self.short_pw2(s)
 
        # Residual add
        out = self.act(s + y)
        return out
