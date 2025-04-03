import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64],
                 kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels] * 2

        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[0],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels[0], out_channels[1],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[1]),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64]):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.trans_conv = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.trans_conv = nn.ConvTranspose3d(in_channels, in_channels,
                                             kernel_size=3, stride=2, padding=1, output_padding=1)
        '''
        nn.ConvTranspose3d(in_channels, in_channels,
                                             kernel_size=3, stride=2, padding=1, output_padding=1)
        '''
        self.double_conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x_trans, x_cat):
        x = self.trans_conv(x_trans)
        #print("x_trans", x_trans.shape)
        #print("x", x.shape)
        #print("x_cat", x_cat.shape)
        x = torch.cat([x_cat, x], dim=1)
        return self.double_conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        #self.bn = nn.BatchNorm3d(out_channels)
        #self.norm = nn.ReLU()
        #self.norm = nn.Tanh()

    def forward(self, x):
        return self.conv(x)


def DSNT_f(h, spacial=None):
    B, C, Z, Y, X = h.shape
    #heatmap = heatmap * 20
    #h = heatmap / torch.sum(heatmap, dim=(2, 3, 4), keepdim=True)
    x = torch.linspace(-1, 1, X).to(h.device)
    y = torch.linspace(-1, 1, Y).to(h.device)
    z = torch.linspace(-1, 1, Z).to(h.device)
    x_cord = x.view([B, 1, 1, 1, X])
    y_cord = y.view([B, 1, 1, Y, 1])
    z_cord = z.view([B, 1, Z, 1, 1])
    px = (h * x_cord).sum(dim=(2, 3)).sum(dim=-1)
    py = (h * y_cord).sum(dim=(2, 4)).sum(dim=-1)
    pz = (h * z_cord).sum(dim=(3, 4)).sum(dim=-1)

    #print(x_cord.shape, px.shape, px.view(B, C, 1, 1, 1).shape)
    var_x = (h * ((x_cord - px.view(B, C, 1, 1, 1)) ** 2)).sum(dim=(2, 3, 4))
    var_y = (h * (y_cord - py.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    var_z = (h * (z_cord - pz.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    return px, py, pz, var_x, var_y, var_z


ch = [16, 32, 64, 128, 256]


class UNet_3D(nn.Module):
    def __init__(self, in_channels, out_classes, last_layer=Out):
        super(UNet_3D, self).__init__()
        self.encoder_1 = DoubleConv(in_channels, [ch[0], ch[1]])  # 16,32
        self.encoder_2 = Down(ch[1], [ch[1], ch[2]])  # 32, 32 64
        self.encoder_3 = Down(ch[2], [ch[2], ch[3]])  # 64,64,128
        self.buttom = Down(ch[3], [ch[3], ch[4]]) # 128,128,256
        self.decoder_3 = Up(ch[4], ch[3])  # 256,128
        self.decoder_2 = Up(ch[3], ch[2])  # 128, 64
        self.decoder_1 = Up(ch[2], ch[1])  # 64, 32
        self.output = last_layer(ch[1], out_classes) if last_layer is not None else None

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e = self.buttom(e3)
        d3 = self.decoder_3(e, e3)
        d2 = self.decoder_2(d3, e2)
        local = self.decoder_1(d2, e1)
        if self.output is not None:
            local = self.output(local)
        norm_local = torch.exp(local) / torch.exp(local).sum(dim=(2, 3, 4), keepdim=True)
        px, py, pz, var_x, var_y, var_z = DSNT_f(norm_local)
        return local, norm_local, px, py, pz, var_x, var_y, var_z


if __name__ == "__main__":
    x = torch.rand(1, 1, 200, 128, 128)
    net = UNet_3D(in_channels=1, out_classes=20)
    # net.train()
    y = net(x)
    print(y.shape)
