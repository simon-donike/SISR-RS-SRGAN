import math
import torch
import torch.nn as nn

# ---------- Core upsampler ----------
def make_upsampler(n_channels, scale):
    stages = []
    for _ in range(int(math.log2(scale))):
        stages += [
            nn.Conv2d(n_channels, n_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        ]
    return nn.Sequential(*stages)

# ---------- Blocks ----------
class ResidualBlockNoBN(nn.Module):
    def __init__(self, n_channels=64, k=3, res_scale=0.2):
        super().__init__()
        p = k // 2
        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, k, padding=p),
            nn.PReLU(),
            nn.Conv2d(n_channels, n_channels, k, padding=p),
        )
        self.res_scale = res_scale
    def forward(self, x):
        return x + self.res_scale * self.body(x)

class RCAB(nn.Module):
    """ Residual Channel Attention Block (no BN) """
    def __init__(self, n_channels=64, k=3, reduction=16, res_scale=0.2):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, k, padding=p)
        self.act   = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, k, padding=p)
        # SE/Channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_channels, n_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // reduction, n_channels, 1),
            nn.Sigmoid()
        )
        self.res_scale = res_scale
    def forward(self, x):
        y = self.conv1(x); y = self.act(y); y = self.conv2(y)
        w = self.se(y)
        return x + self.res_scale * (y * w)

class DenseBlock5(nn.Module):
    """ ESRGAN-style dense block (5 convs) """
    def __init__(self, nf=64, gc=32, k=3):
        super().__init__()
        p = k // 2
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.c1 = nn.Conv2d(nf,      gc, k, padding=p)
        self.c2 = nn.Conv2d(nf+gc,   gc, k, padding=p)
        self.c3 = nn.Conv2d(nf+2*gc, gc, k, padding=p)
        self.c4 = nn.Conv2d(nf+3*gc, gc, k, padding=p)
        self.c5 = nn.Conv2d(nf+4*gc, nf, k, padding=p)
    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5

class RRDB(nn.Module):
    """ Residual-in-Residual Dense Block """
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.db1 = DenseBlock5(nf, gc)
        self.db2 = DenseBlock5(nf, gc)
        self.db3 = DenseBlock5(nf, gc)
        self.res_scale = res_scale
    def forward(self, x):
        y = self.db1(x)
        y = self.db2(y)
        y = self.db3(y)
        return x + self.res_scale * y

class LKA(nn.Module):
    """
    Large-Kernel Attention (lightweight): DW 5x5 -> dilated DW 7x7(d=3) -> 1x1 mixing
    Good for 8× to increase RF without downsampling.
    """
    def __init__(self, n_channels=64):
        super().__init__()
        self.dw5  = nn.Conv2d(n_channels, n_channels, 5, padding=2, groups=n_channels)
        self.dw7d = nn.Conv2d(n_channels, n_channels, 7, padding=9, dilation=3, groups=n_channels)
        self.pw   = nn.Conv2d(n_channels, n_channels, 1)
    def forward(self, x):
        attn = self.dw5(x)
        attn = self.dw7d(attn)
        attn = self.pw(attn)
        return x * torch.sigmoid(attn)

class LKAResBlock(nn.Module):
    def __init__(self, n_channels=64, k=3, res_scale=0.2):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, k, padding=p)
        self.act   = nn.PReLU()
        self.lka   = LKA(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, k, padding=p)
        self.res_scale = res_scale
    def forward(self, x):
        y = self.conv1(x); y = self.act(y)
        y = self.lka(y)
        y = self.conv2(y)
        return x + self.res_scale * y

# ---------- Flexible generator ----------
class SRResNet_NoBN_Flex(nn.Module):
    """
    Drop-in SR generator for 2/4/8× with selectable block type:
      block_type ∈ {"res", "rcab", "rrdb", "lka"}
    - No BN, residual scaling, linear output head.
    """
    def __init__(self, in_channels=6, n_channels=96, n_blocks=32,
                 small_kernel=3, large_kernel=9, scale=8,
                 block_type="rcab"):
        super().__init__()
        assert scale in {2,4,8}
        self.scale = scale

        # head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, large_kernel, padding=large_kernel//2),
            nn.PReLU(),
        )

        # choose body block
        if block_type == "res":
            Block = lambda: ResidualBlockNoBN(n_channels, small_kernel, res_scale=0.2)
        elif block_type == "rcab":
            Block = lambda: RCAB(n_channels, small_kernel, reduction=16, res_scale=0.2)
        elif block_type == "rrdb":
            Block = lambda: RRDB(nf=n_channels, gc=max(16, n_channels//3), res_scale=0.2)
        elif block_type == "lka":
            Block = lambda: LKAResBlock(n_channels, small_kernel, res_scale=0.2)
        else:
            raise ValueError("block_type must be one of {'res','rcab','rrdb','lka'}")

        self.body = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(n_channels, n_channels, small_kernel, padding=small_kernel//2)

        # upsampler
        self.upsampler = make_upsampler(n_channels, scale)

        # tail (no activation)
        self.tail = nn.Conv2d(n_channels, in_channels, large_kernel, padding=large_kernel//2)

    def forward(self, x):
        fea = self.head(x)
        res = self.body(fea)
        res = self.body_tail(res)
        fea = fea + res
        fea = self.upsampler(fea)
        out = self.tail(fea)
        return out


if __name__ == '__main__':
    # test out models
    model = SRResNet_NoBN_Flex(
    in_channels=6, scale=8,
    n_channels=96, n_blocks=32,
    block_type="rrdb"      # try "rrdb" or "lka" next
    )
    #count model parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    # test model
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(y.shape)  # (1, 6, 512, 512)


