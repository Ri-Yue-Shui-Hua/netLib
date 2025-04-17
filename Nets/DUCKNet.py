'''
Author: wmz 1217849284@qq.com
Date: 2025-04-17 17:06:56
LastEditors: wmz 1217849284@qq.com
LastEditTime: 2025-04-17 17:07:27
FilePath: \netLib\Nets\DUCKNet.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust mid_channels here after concatenation to handle the input channels correctly
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, in_channels)  # in_channels + in_channels // 2 because of concatenation
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handling the size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # This causes the channel count to double
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DUCKNet(nn.Module):
    def __init__(self, img_height, img_width, input_channels, out_classes, starting_filters, bilinear=True):
        super(DUCKNet, self).__init__()
        self.inc = DoubleConv(input_channels, starting_filters)
        self.down1 = Down(starting_filters, starting_filters * 2)
        self.down2 = Down(starting_filters * 2, starting_filters * 4)
        self.down3 = Down(starting_filters * 4, starting_filters * 8)
        self.down4 = Down(starting_filters * 8, starting_filters * 16)
        self.down5 = Down(starting_filters * 16, starting_filters * 32)

        self.up1 = Up(starting_filters * 32, starting_filters * 16, bilinear)
        self.up2 = Up(starting_filters * 16, starting_filters * 8, bilinear)
        self.up3 = Up(starting_filters * 8, starting_filters * 4, bilinear)
        self.up4 = Up(starting_filters * 4, starting_filters * 2, bilinear)
        self.up5 = Up(starting_filters * 2, starting_filters, bilinear)

        self.outc = OutConv(starting_filters, out_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits