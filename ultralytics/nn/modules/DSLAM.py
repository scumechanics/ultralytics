import torch
import torch.nn as nn
 
 
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
 
    def forward(self, x):
        return self.fn(x) + x
 
 
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
 
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)  # (b, heads, n, dim_head)
        k = self.to_k(x).view(b, n, h, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, self.dim_head).transpose(1, 2)
 
        q = torch.nn.functional.softmax(q, dim=-1)
        k = torch.nn.functional.softmax(k, dim=-1)
 
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
 
 
class DSLAM(nn.Module):
    def __init__(self, c1, n=1, reduction=16, heads=8, dim_head=64):
        super(DSLAM, self).__init__()
        c2 = c1
 
        # 多尺度卷积操作
        self.DCovN = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Sequential(
                    nn.Conv2d(c2, c2, kernel_size=5, padding=2, groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(1, 7), padding=(0, 3), groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(7, 1), padding=(3, 0), groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(1, 11), padding=(0, 5), groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(11, 1), padding=(5, 0), groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(1, 21), padding=(0, 10), groups=c2),
                    nn.Conv2d(c2, c2, kernel_size=(21, 1), padding=(10, 0), groups=c2),
                ),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
 
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )
 
        self.linear_attention = LinearAttention(c2, heads, dim_head)
 
        self._initialize_weights()
        self.initialize_layer(self.fc)
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        y = x * y.expand_as(x)
 
        b, c, h, w = y.size()
        y = y.view(b, c, -1).transpose(1, 2)  # (b, n, c)
        y = self.linear_attention(y)
        y = y.transpose(1, 2).view(b, c, h, w)
 
        return y
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
 
 
