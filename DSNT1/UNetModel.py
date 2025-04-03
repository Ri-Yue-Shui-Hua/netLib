from nets.UNetParts import *
import torch.nn.functional as F
from torch import nn
import torch


class DSNT(object):
    def __init__(self, heatmap, norm_method='no_softmax'):
        super(DSNT, self).__init__()
        self.heatmap = heatmap
        self.x_len = heatmap.shape[4]
        self.y_len = heatmap.shape[3]
        self.z_len = heatmap.shape[2]
        self.x_weight = self.gen_weight('x')
        self.y_weight = self.gen_weight('y')
        self.z_weight = self.gen_weight('z')
        self.norm_method = norm_method

    def gen_weight(self, t):
        if t == 'x':
            length = self.x_len
            new_shape = [1, 1, 1, 1, length]
        elif t == 'y':
            length = self.y_len
            new_shape = [1, 1, 1, length, 1]
        elif t == 'z':
            length = self.z_len
            new_shape = [1, 1, length, 1, 1]
        else:
            raise ValueError("Error, No such type of Weight")

        #n = (length - 1) / length
        #w = torch.linspace(-n, n, length)
        w = torch.linspace(-1, 1, length, requires_grad=False)
        w = w.view(new_shape)
        return w.cuda(0)

    def forward(self):
        B, C, Z, Y, X = self.heatmap.shape
        #print(self.heatmap.shape, X, Y, Z)
        x_coords = torch.zeros([B, C])
        y_coords = torch.zeros([B, C])
        z_coords = torch.zeros([B, C])
        var_x_coords = torch.zeros([B, C])
        var_y_coords = torch.zeros([B, C])
        var_z_coords = torch.zeros([B, C])
        bias = torch.Tensor([1]).cuda(0)
        torch.nn.init.constant_(bias, 0)
        for bz in range(B):
            for ch in range(C):
                h = self.heatmap[bz, ch, :]
                #print(h.min(), h.max())
                h = h ** 7
                h = h / h.sum(())
                #print(h.min(), h.max())
                if self.norm_method == 'softmax':
                    h = torch.exp(h) / torch.exp(h).sum()
                else:
                    h = h / h.sum()
                h.unsqueeze_(0)
                h.unsqueeze_(0)
                x = F.conv3d(h, self.x_weight, bias=bias, stride=1)
                y = F.conv3d(h, self.y_weight, bias=bias, stride=1)
                z = F.conv3d(h, self.z_weight, bias=bias, stride=1)
                x_coords[bz, ch] = x.sum()
                y_coords[bz, ch] = y.sum()
                z_coords[bz, ch] = z.sum()
                var_x = F.conv3d(h, (self.x_weight - x.sum()) ** 2, bias=bias, stride=1)
                var_y = F.conv3d(h, (self.y_weight - y.sum()) ** 2, bias=bias, stride=1)
                var_z = F.conv3d(h, (self.z_weight - z.sum()) ** 2, bias=bias, stride=1)
                var_x_coords[bz, ch] = var_x.sum()
                var_y_coords[bz, ch] = var_y.sum()
                var_z_coords[bz, ch] = var_z.sum()
        return x_coords, y_coords, z_coords, var_x_coords, var_y_coords, var_z_coords


class cord_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class coord_layer(nn.Module):
    def __init__(self):
        super(coord_layer, self).__init__()
        self.conv1 = cord_OutConv(18, 18*3).cuda(0)
        self.conv2 = cord_OutConv(18*3, 18*3).cuda(0)
        self.gap = nn.AdaptiveAvgPool3d(1).cuda(0)

    def forward(self, h):
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.gap(h)
        h = h.view([-1, 18, 3]).cuda(0)
        #print(h, h.shape)
        x = h[:, :, 0]
        y = h[:, :, 1]
        z = h[:, :, 2]
        return x, y, z

def DSNT_f(heatmap, spacial):
    B, C, Z, Y, X = heatmap.shape
    #heatmap = heatmap * 20
    h = torch.exp(heatmap) / torch.exp(heatmap).sum(dim=(2, 3, 4), keepdim=True)
    h = h * spacial
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

class UNet(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.conv = DoubleConv(channels, 16, 32).cuda(0)
        self.down1 = Down(32, 32, 64).cuda(0)
        self.down2 = Down(64, 64, 128).cuda(0)
        self.down3 = Down(128, 128, 256).cuda(0)

        self.up1 = Up(384, 128, 128).cuda(0)
        self.up2 = Up(192, 64, 64).cuda(0)
        self.up3 = Up(96, 32, 32).cuda(0)
        self.out = OutConv(32, classes).cuda(0)
        self.spacial = Spacial_Info(32, classes).cuda(0)
        #self.cord = coord_layer().cuda(0)
        # self.norm = nn.ReLU(inplace=True)
        #self.cord = DSNT

        # self.edge = EdgeOut(32, 26).cuda(1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = x3.cuda(0)
        x2 = x2.cuda(0)
        x1 = x1.cuda(0)
        x0 = x0.cuda(0)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        # edge = self.edge(x)

        local = self.out(x)

        spacial = self.spacial(x)
        final = local * spacial
        #dsnt_func = DSNT(final)
        #x, y, z = dsnt_func.forward()
        x, y, z, vx, vy, vz = DSNT_f(local, spacial)
        # for check
        # self.localt = local
        # self.spacialt = spacial
        # x = self.norm(x)
        return final, spacial, local, x.cuda(0), y.cuda(0), z.cuda(0), vx.cuda(0), vy.cuda(0), vz.cuda(0)
        # return [x, edge]


class SCN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(channels=1, classes=15)
        self.sccov0 = nn.Conv3d(in_channels=15, out_channels=64, kernel_size=[7, 7, 7], padding=3).cuda(1)
        self.sccov1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[7, 7, 7], padding=3).cuda(1)
        self.sccov2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[7, 7, 7], padding=3).cuda(1)
        self.sccov3 = nn.Conv3d(in_channels=64, out_channels=15, kernel_size=[7, 7, 7], padding=3).cuda(1)
        self.downsample = nn.MaxPool3d(kernel_size=4).cuda(0)
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True).cuda(1)

    def forward(self, x):
        local_pred = self.unet(x).cuda(1)
        spatial_pred = self.downsample(local_pred)
        spatial_pred = self.sccov0(spatial_pred)
        spatial_pred = self.sccov1(spatial_pred)
        spatial_pred = self.sccov2(spatial_pred)
        spatial_pred = self.sccov3(spatial_pred)
        spatial_pred = self.upsample(spatial_pred)
        prediction = spatial_pred * local_pred
        return prediction


if __name__ == '__main__':
    heatmap = torch.zeros((7, 7, 7))
    heatmap[1, 1, 1] = 4
    heatmap[1, 1, 2] = 3
    heatmap[1, 1, 3] = 2
    heatmap[1, 0, 1] = 1
    heatmap = heatmap / heatmap.sum()
    heatmap.unsqueeze_(0)
    heatmap.unsqueeze_(0)
    heatmap = heatmap.cuda(0)
    dsnt = DSNT(heatmap)
    z, y, x = dsnt.forward()
    print(x, y, z)
    def get_cord(x):
        return (x+1)*7/2


    x_cord, y_cord, z_cord = get_cord(x), get_cord(y), get_cord(z)
    print(x_cord, y_cord, z_cord)
    print(heatmap[0, 0, int(x_cord), int(y_cord), int(z_cord)])
    print(heatmap[0, 0, 1, 1, 1])

