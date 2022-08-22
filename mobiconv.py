import torch
import torch.nn as nn
import torch.nn.functional as F


class MobiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=True, n_layers=4, ratio=0.1):
        super(MobiConvBlock, self).__init__()
        # out_channels should be divisible by n_layers
        self.n_layers = n_layers
        self.ratio = ratio
        assert out_channels % n_layers == 0

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels // n_layers, kernel_size=kernel_size,
                          padding=padding, stride=stride, bias=bias)
            )

    def forward(self, x):
        size = 2 ** (self.n_layers - 1)
        out = []
        table = torch.ones_like(x[:, :x.shape[1] // self.n_layers, :, :])
        for conv in self.convs:
            h = F.avg_pool2d(x, kernel_size=size, stride=size)
            h = conv(h)
            h = F.upsample(h, scale_factor=size, mode='nearest')
            h = table * h + torch.logical_not(table) * conv.bias.unsqueeze(1).unsqueeze(2)
            threshold = self.ratio * torch.amax(h, dim=(-2, -1), keepdim=True)
            table = torch.ge(h, threshold)
            out.append(h)
            size //= 2
        out = torch.cat(out, dim=1)
        return out


class SmartPool2d(nn.Module):
    def __init__(self, scale, mode='avgpool', ratio=0.1):
        super(SmartPool2d, self).__init__()
        self.scale = scale
        self.mode = mode
        self.ratio = ratio

    def _crop(self, x):
        H, W = x.shape
        threshold = self.ratio * torch.amax(x, dim=(-2, -1))
        _x = [H - 1, 0]
        _y = [W - 1, 0]
        for i in range(H):
            for j in range(W):
                if x[i, j] >= threshold:
                    if i <= _x[0]:
                        _x[0] = i
                    elif i >= _x[1]:
                        _x[1] = i
                    if j <= _y[0]:
                        _y[0] = j
                    elif j >= _y[1]:
                        _y[1] = j
        return x[_x[0]:_x[1] + 1, _y[0]:_y[1] + 1], _x, _y

    def forward(self, x):
        N, C, H, W = x.shape
        out = []
        for n in range(N):
            stack = []
            for c in range(C):
                feature, _x, _y = self._crop(x[n, c, :, :])
                if feature.shape[-2] == 0 or feature.shape[-1] == 0:
                    print(_x)
                    print(_y)
                if self.mode == 'avgpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H // self.scale, W // self.scale),
                                            align_corners=False, antialias=True, mode='bilinear')
                elif self.mode == 'maxpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H, W),
                                            align_corners=False, antialias=True, mode='bilinear')
                    feature = F.max_pool2d(feature, kernel_size=self.scale, stride=self.scale)
                stack.append(feature.squeeze(1).squeeze(0))
            stack = torch.stack(stack, dim=0)
            out.append(stack)
        out = torch.stack(out, dim=0)
        return out