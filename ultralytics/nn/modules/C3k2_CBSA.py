# https://github.com/QishuaiWen/CBSA
import torch
from torch import nn
from einops import rearrange, repeat

class TSSA(nn.Module):
    # https://github.com/RobinWu218/ToST/blob/main/tost_vision/tost.py
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        num_heads = heads
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.qkv = nn.Linear(dim, dim, bias=False)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Linear(dim, dim)
        self.scale = dim_head ** -0.5

    def forward(self, x, return_attn=False):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        b, h, N, d = w.shape
        if return_attn:
            dots = w @ w.transpose(-1, -2)
            return self.attend(dots)

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2
        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}


class MSSA(nn.Module):
    # https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate.py
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, return_attn=False):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        if return_attn:
            return attn
        out = torch.matmul(attn, w)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CBSA(nn.Module):
    """
    Cross-Block Self-Attention module.
    Adapted to work with 2D feature maps (B, C, H, W) instead of sequences.
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, inner_dim, bias=False)

        self.step_x = nn.Parameter(torch.randn(heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(heads, 1, 1))

        self.to_out = nn.Linear(inner_dim, dim)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        self.qkv = nn.Identity()

    def attention(self, query, key, value):
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ value
        return out, attn

    def forward(self, x, return_attn=False):
        """
        Forward pass for CBSA.
        
        Args:
            x: Input tensor of shape (B, C, H, W) - 2D feature map
            return_attn: Whether to return attention weights
        
        Returns:
            Output tensor of shape (B, C, H, W) - 2D feature map
        """
        b, c, h, w = x.shape
        width = w  # avoid name collision with projected tensor
        n = h * w
        inner_dim = self.heads * self.dim_head
        
        # Convert 2D feature map to sequence format: (B, C, H, W) -> (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Project to inner dimension
        proj = self.proj(x_seq)  # (B, n, inner_dim)
        self.qkv(proj)
        
        # Create representation tokens using pooling
        # Use full feature map to avoid shape mismatch; pool to fixed 8x8 tokens
        if n > 1:
            proj_2d = proj.reshape(b, h, width, inner_dim).permute(0, 3, 1, 2)  # (B, inner_dim, h, w)
            rep = self.pool(proj_2d)  # (B, inner_dim, 8, 8)
            rep = rep.reshape(b, inner_dim, -1).permute(0, 2, 1)  # (B, 64, inner_dim)
        else:
            # Handle edge case when H*W = 1
            rep = proj.reshape(b, 1, inner_dim).repeat(1, 64, 1)  # (B, 64, inner_dim) - repeat single token
        
        # Reshape for attention
        proj = proj.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, n, dim_head)
        rep = rep.reshape(b, 64, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, 64, dim_head)
        
        # Cross attention: rep attends to w
        rep_delta, attn = self.attention(rep, proj, proj)

        if return_attn:
            return attn.transpose(-1, -2) @ attn

        # Update representation
        rep = rep + self.step_rep * rep_delta

        # Self attention on representation
        x_delta, _ = self.attention(rep, rep, rep)
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta

        # Reshape back to sequence: (B, heads, n, dim_head) -> (B, n, heads*dim_head)
        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')
        x_out = self.to_out(x_delta)
        
        # Convert back to 2D feature map: (B, H*W, C) -> (B, C, H, W)
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=h, w=w)
        
        return x_out


 
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k_CBSA(C3k):
    """C3k module with CBSA attention blocks."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, heads=8, dim_head=64):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBSA(c_, heads=heads, dim_head=dim_head) for _ in range(n)))


class C3k2_CBSA(C3k2):
    """C3k2 module with CBSA attention blocks."""
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, heads=8, dim_head=64):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_CBSA(self.c, self.c, 2, shortcut, g, heads=heads, dim_head=dim_head) if c3k 
            else CBSA(self.c, heads=heads, dim_head=dim_head) 
            for _ in range(n)
        )
