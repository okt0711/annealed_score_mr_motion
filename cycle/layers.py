import torch
import torch.nn as nn
import functools


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = False
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
        use_bias = True
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer, use_bias


def get_act_layer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU
    elif act_type == 'lrelu':
        act_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2)
    elif act_type == 'sigmoid':
        act_layer = nn.Sigmoid
    elif act_type == 'none':
        def act_layer(): return Identity()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(ConvBlock, self).__init__()
        norm_layer1, use_bias1 = get_norm_layer(norm_type)
        act_layer1 = get_act_layer(act_type)
        norm_layer2, use_bias2 = get_norm_layer(norm_type)
        act_layer2 = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias1),
            norm_layer1(out_channels),
            act_layer1(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias2),
            norm_layer2(out_channels),
            act_layer2()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels, out_channels, norm_type, act_type)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(UpBlock, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.layers2 = ConvBlock(in_channels, out_channels, norm_type, act_type)

    def forward(self, x1, x2):
        x1 = self.layers1(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.layers2(x)
        return out