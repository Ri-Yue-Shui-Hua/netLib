'''
Author: wmz 1217849284@qq.com
Date: 2025-04-17 18:13:59
LastEditors: wmz 1217849284@qq.com
LastEditTime: 2025-04-17 18:22:26
FilePath: \netLib\weight_init.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch.nn as nn


def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def gaussian_weight_init(m, conv_std=0.01, bn_std=0.01):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        m.weight.data.normal_(0, conv_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()