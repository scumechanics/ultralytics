import torch.nn as nn
import torch
 
 
# Helper function to handle padding
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
# Basic Convolution Block
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
 
# Cross-Level Channel Attention (CLCA) Module
class CLCA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11, level_factor=2):
        super().__init__()
 
        # Cross-level attention mechanism
        self.avg_pool = nn.AvgPool2d(7, 1, 3)  # Initial average pooling to capture global context
        self.conv1 = Conv(ch, ch)  # First convolution to process features
 
        # Horizontal and Vertical Convolutions (extracting directional context)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
 
        # Attention across different levels (local and global features)
        self.level_factor = level_factor  # Number of levels to aggregate across
        self.level_convs = nn.ModuleList([Conv(ch, ch) for _ in range(level_factor)])  # Multi-level convolutions
 
        # Final convolution to merge features
        self.merge_conv = Conv(ch * level_factor, ch)  # Merge the level features into a single tensor
 
        # Final convolution to compute attention
        self.conv2 = Conv(ch, ch)
 
        # Activation function
        self.act = nn.Sigmoid()
 
    def forward(self, x):
        # Process each level of features separately
        level_outputs = []
        for i in range(self.level_factor):
            # Each level gets its own convolutional processing path
            level_features = self.conv1(x)
            level_features = self.level_convs[i](level_features)
 
            # Horizontal and Vertical convolutions at each level
            level_features = self.h_conv(level_features)
            level_features = self.v_conv(level_features)
            level_outputs.append(level_features)
 
        # Combine level outputs (concatenate along the channel dimension)
        combined_features = torch.cat(level_outputs, dim=1)  # Concatenate along the channel dimension
 
        # Apply the merge convolution to adjust the number of channels after concatenation
        merged_features = self.merge_conv(combined_features)  # Adjust to the correct channel size
 
        # Apply attention to the input features
        attn_factor = self.act(self.conv2(merged_features))  # Apply attention computation
 
        # Apply attention to the input features
        return attn_factor * x  # Return the weighted features
 
 
class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.
    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.
    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.
    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
 
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()
 
        self.attn = CLCA(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut
 
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
 
 
class C2PSA_CLCA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.
    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """
 
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
 
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
 
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
 
