import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np


def cal_img_cord(output):
    N, C, Z, Y, X = output.shape
    cord_list = []
    for i in range(N):
        for j in range(C):
            torch_out = torch.exp(output[i][j]) / torch.sum(torch.exp(output[i][j]))
            np_out_arr = torch_out.cpu().numpy()
            re = np.where(np_out_arr == np_out_arr.max())
            # Image(Z, Y, X) ==  Slicer(R, A, S)
            # Image Matrix coordinate is (z , y , x)
            # Here, return the (x, y, z ) with regard to Slicer Coordinate
            cord_list.append([re[2][0], re[1][0], re[0][0]])
    return cord_list


def get_argmax_cords(heatmap):
    C, Z_len, Y_len, X_len = heatmap.shape
    heatmap = torch.from_numpy(heatmap)
    heatmap.unsqueeze_(0)
    # cord_list 18*3, WRT. Slicer
    cord_list = cal_img_cord(heatmap)
    cords = np.array(cord_list)
    print('GT:\t', cords)
    x_cord = cords[:, 0]
    y_cord = cords[:, 1]
    z_cord = cords[:, 2]
    def norm(x, x_len):
        # (x-mean)/mean
        return (2*x - x_len) / x_len
        #return x/x_len
    x, y, z = norm(cords[:, 0], X_len), norm(cords[:, 1], Y_len), norm(cords[:, 2], Z_len)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    z = torch.from_numpy(z).float()
    return x, y, z, x_cord, y_cord, z_cord

def DSNT_f(heatmap):
    B, C, Z, Y, X = heatmap.shape
    heatmap = heatmap * 90
    h = torch.exp(heatmap) / torch.exp(heatmap).sum(dim=(2, 3, 4), keepdim=True)
    #h = heatmap / torch.sum(heatmap, dim=(2, 3, 4), keepdim=True)
    x = torch.linspace(-1, 1, X)
    y = torch.linspace(-1, 1, Y)
    z = torch.linspace(-1, 1, Z)
    x_cord = x.view([B, C, 1, 1, X])
    y_cord = y.view([B, C, 1, Y, 1])
    z_cord = z.view([B, C, Z, 1, 1])
    px = (h * x_cord).sum(dim=(2, 3)).sum(dim=-1)
    py = (h * y_cord).sum(dim=(2, 4)).sum(dim=-1)
    pz = (h * z_cord).sum(dim=(3, 4)).sum(dim=-1)
    print(x_cord.shape, px.shape, px.view(B, C, 1, 1, 1).shape)
    var_x = (h * (x_cord - px.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    var_y = (h * (y_cord - py.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    var_z = (h * (z_cord - pz.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))

    return px, py, pz, var_x, var_y, var_z



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
                print(h.min(), h.max())
                h = h ** 97
                print(h.min(), h.max())
                if self.norm_method == 'softmax':
                    h = torch.exp(h) / torch.exp(h).sum()
                else:
                    h = h / h.sum()
                print(h.min(), h.max())
                h.unsqueeze_(0)
                h.unsqueeze_(0)
                x = F.conv3d(h, self.x_weight, bias=bias, stride=1)
                y = F.conv3d(h, self.y_weight, bias=bias, stride=1)
                z = F.conv3d(h, self.z_weight, bias=bias, stride=1)
                x_coords[bz, ch] = x.sum()
                y_coords[bz, ch] = y.sum()
                z_coords[bz, ch] = z.sum()

                var_x = h * (self.x_weight - x.sum()) ** 2
                var_y = h * (self.y_weight - y.sum()) ** 2
                var_z = h * (self.z_weight - z.sum()) ** 2
                var_x = F.conv3d(h, (self.x_weight - x.sum()) ** 2, bias=bias, stride=1)
                var_y = F.conv3d(h, (self.y_weight - y.sum()) ** 2, bias=bias, stride=1)
                var_z = F.conv3d(h, (self.z_weight - z.sum()) ** 2, bias=bias, stride=1)
                var_x_coords[bz, ch] = var_x.sum()
                var_y_coords[bz, ch] = var_y.sum()
                var_z_coords[bz, ch] = var_z.sum()
        return x_coords, y_coords, z_coords, var_x_coords, var_y_coords, var_z_coords


if __name__ == "__main__":
    heatmap_path = r"E:\Data\HipLandmark\Heatmap_2\CTPEL\0001\1.nii.gz"
    sitk_heatmap = sitk.ReadImage(heatmap_path)
    heatmap_arr = sitk.GetArrayFromImage(sitk_heatmap)
    heatmap_arr = heatmap_arr[np.newaxis, np.newaxis, :]
    heatmap_arr = torch.from_numpy(heatmap_arr).float()
    heatmap_arr = heatmap_arr
    #h = torch.exp(heatmap_arr) / torch.exp(heatmap_arr).sum()

    #dsnt = DSNT(heatmap_arr, 'no_softmax')
    #x, y, z , var_x, var_y, var_z = dsnt.forward()
    x, y, z, var_x, var_y, var_z = DSNT_f(heatmap_arr)
    gx, gy, gz, gx_cord, gy_cord, gz_cord = get_argmax_cords(heatmap_arr[0].cpu().numpy())
    print('DSNT output\t', x, y, z)
    print('Ground Truth\t', gx, gy, gz)
    print('Var', var_x, var_y, var_z)
    for a, b, c in zip(x[0], y[0], z[0]):
        px = (a + 1) * 200 / 2
        py = (b + 1) * 128 / 2
        pz = (c + 1) * 128 / 2
        p_mre = (a-gx)**2 + (b-gy)**2 + (c-gz)**2
        p_mre = p_mre.mean()
        c_mre = (px - gx_cord) ** 2 + (py - gy_cord) ** 2 + (pz - gz_cord) ** 2
        c_mre = c_mre.mean()
        print('*' * 10, px, py, pz, '*' * 10)
        print('#' * 10, a, b, c, '#' * 10)
        print('P_MRE', p_mre)
        print('C_MRE', c_mre)