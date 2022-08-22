import torch
import torch.nn as nn
import torch.nn.functional as F


class MobiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=True, n_layers=4, ratio=0.1):
        super(MobiConvBlock, self).__init__()
        # out_channels should be divisible by n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        N, C, H, W = x.shape
        size = 2 ** (self.n_layers - 1)
        out = []
        table = torch.ones(N, self.out_channels // self.n_layers, H, W).cuda()
        for conv in self.convs:
            h = F.avg_pool2d(x, kernel_size=size, stride=size)
            h = conv(h)
            h = F.upsample(h, scale_factor=size, mode='nearest')
            h = table * h + torch.logical_not(table) * conv.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
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
        N, C, H, W = x.shape
        threshold = self.ratio * torch.amax(x, dim=(-2, -1))
        table = torch.ge(x, threshold.unsqueeze(2).unsqueeze(3))
        x_range = torch.tile(torch.arange(H), (N, C, W, 1)).permute(0, 1, 3, 2).cuda()
        y_range = torch.tile(torch.arange(W), (N, C, H, 1)).cuda()
        x_min = torch.amin(torch.logical_not(x_range) * 1e5 + table * x_range, dim=(-2, -1))
        x_max = torch.amax(torch.logical_not(x_range) * -1e5 + table * x_range, dim=(-2, -1))
        y_min = torch.amin(torch.logical_not(y_range) * 1e5 + table * y_range, dim=(-2, -1))
        y_max = torch.amax(torch.logical_not(y_range) * -1e5 + table * y_range, dim=(-2, -1))
        out = []
        for n in range(N):
            stack = []
            for c in range(C):
                feature = x[n, c, int(x_min[n, c].item()):int(x_max[n, c].item()) + 1,
                            int(y_min[n, c].item()):int(y_max[n, c].item()) + 1]
                print(feature.shape)
                stack.append(feature)
            out.append(torch.stack(stack, dim=1))
        return torch.stack(out, dim=0)

    def forward(self, x):
        N, C, H, W = x.shape
        threshold = self.ratio * torch.amax(x, dim=(-2, -1))
        table = torch.ge(x, threshold.unsqueeze(2).unsqueeze(3))
        x_range = torch.tile(torch.arange(H), (N, C, W, 1)).permute(0, 1, 3, 2).cuda()
        y_range = torch.tile(torch.arange(W), (N, C, H, 1)).cuda()
        x_min = torch.amin(torch.logical_not(x_range) * 1e5 + table * x_range, dim=(-2, -1))
        x_max = torch.amax(torch.logical_not(x_range) * -1e5 + table * x_range, dim=(-2, -1))
        y_min = torch.amin(torch.logical_not(y_range) * 1e5 + table * y_range, dim=(-2, -1))
        y_max = torch.amax(torch.logical_not(y_range) * -1e5 + table * y_range, dim=(-2, -1))
        out = []
        for n in range(N):
            stack = []
            for c in range(C):
                feature = x[n, c, int(x_min[n, c].item()):int(x_max[n, c].item()) + 1,
                          int(y_min[n, c].item()):int(y_max[n, c].item()) + 1]
                if self.mode == 'avgpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H // self.scale, W // self.scale),
                                            align_corners=False, antialias=True, mode='bilinear')
                elif self.mode == 'maxpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H, W),
                                            align_corners=False, antialias=True, mode='bilinear')
                    feature = F.max_pool2d(feature, kernel_size=self.scale, stride=self.scale)
                feature = feature.squeeze(1).squeeze(0)
                stack.append(feature)
            out.append(torch.stack(stack, dim=0))
        out = torch.stack(out, dim=0)
        return out