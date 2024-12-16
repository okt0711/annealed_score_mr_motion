import torch
import torch.nn as nn
import bsa.layers as layers

ConvBlock = layers.ConvBlock
DownBlock = layers.DownBlock
UpBlock = layers.UpBlock

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(Unet, self).__init__()
        self.down1 = ConvBlock(in_channels, opt.ngf, opt.norm_type, opt.act_type)
        self.down2 = DownBlock(opt.ngf, opt.ngf * 2, opt.norm_type, opt.act_type)
        self.down3 = DownBlock(opt.ngf * 2, opt.ngf * 4, opt.norm_type, opt.act_type)
        self.down4 = DownBlock(opt.ngf * 4, opt.ngf * 8, opt.norm_type, opt.act_type)
        self.bridge = DownBlock(opt.ngf * 8, opt.ngf * 16, opt.norm_type, opt.act_type)
        self.up4 = UpBlock(opt.ngf * 16, opt.ngf * 8, opt.norm_type, opt.act_type)
        self.up3 = UpBlock(opt.ngf * 8, opt.ngf * 4, opt.norm_type, opt.act_type)
        self.up2 = UpBlock(opt.ngf * 4, opt.ngf * 2, opt.norm_type, opt.act_type)
        self.up1 = UpBlock(opt.ngf * 2, opt.ngf, opt.norm_type, opt.act_type)
        self.conv1x1 = nn.Conv2d(opt.ngf, out_channels, kernel_size=1, stride=1)
        self.last_conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        br = self.bridge(d4)
        u4 = self.up4(br, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        c1 = self.conv1x1(u1)
        res = c1 + x
        cc = torch.cat([c1, res], dim=1)
        out = self.last_conv(cc)
        return out
