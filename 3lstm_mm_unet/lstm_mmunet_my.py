import sys
sys.path.append("..")


from src.utils import *
from src.backbone import *

import torch
import torch.nn as nn

import math
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MMUnet(nn.Module):
    """Multi-Modal-Unet"""
    def __init__(self, input_nc, output_nc=5, ngf=32):
        super(MMUnet, self).__init__()
        print('~' * 50)
        print(' ----- Creating MULTI_UNET  ...')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1) Flair 1
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = maxpool()
        self.down_2_0 = ConvBlock2d(self.out_dim * 2, self.out_dim * 2)
        self.pool_2_0 = maxpool()
        self.down_3_0 = ConvBlock2d(self.out_dim * 6, self.out_dim * 4)
        self.pool_3_0 = maxpool()
        self.down_4_0 = ConvBlock2d(self.out_dim * 14, self.out_dim * 8)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2) T1
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = maxpool()
        self.down_2_1 = ConvBlock2d(self.out_dim * 2, self.out_dim * 2)
        self.pool_2_1 = maxpool()
        self.down_3_1 = ConvBlock2d(self.out_dim * 6, self.out_dim * 4)
        self.pool_3_1 = maxpool()
        self.down_4_1 = ConvBlock2d(self.out_dim * 14, self.out_dim * 8)
        self.pool_4_1 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = ConvBlock2d(self.out_dim * 30, self.out_dim * 16)

        # ~~~ Decoding Path ~~~~~~ #

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

        self.sa = SpatialAttention()
        self.ca = ChannelAttention(512)
        self.ca1 = ChannelAttention(256)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(64)
        self.ca4 = ChannelAttention(32)

    def forward(self, input):
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = input[:, 0:1, :, :]   # bz * 1  * height * width
        i1 = input[:, 1:2, :, :]
        # print("i0.shape", i0.shape)
        # print("i1.shape", i1.shape)

        # i0 = self.sa(i0) * i0
        # i1 = self.sa(i1) * i1

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # bz * outdim * height * width
        down_1_1 = self.down_1_1(i1)
        # print("down_1_0.shape", down_1_0.shape)
        # print("down_1_1.shape", down_1_1.shape)

        # -----  Second Level --------
        # Batch Size * (outdim * 4) * (volume_size/2) * (height/2) * (width/2)
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_0(down_1_0)), dim=1)
        # print("input_2nd_0.shape", input_2nd_0.shape)
        # print("input_2nd_1.shape", input_2nd_1.shape)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        # print("down_2_0.shape", down_2_0.shape)
        # print("down_2_1.shape", down_2_1.shape)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        # print("down_2_0m.shape", down_2_0m.shape)
        # print("down_2_1m.shape", down_2_1m.shape)

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)
        # print("input_3rd_0.shape", input_3rd_0.shape)
        # print("input_3rd_1.shape", input_3rd_1.shape)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        # print("down_3_0.shape", down_3_0.shape)
        # print("down_3_1.shape", down_3_1.shape)

        # -----  Fourth Level --------
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_0(down_3_1)
        # print("down_3_0m.shape", down_3_0m.shape)
        # print("down_3_1m.shape", down_3_1m.shape)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)
        # print("input_4th_0.shape", input_4th_0.shape)
        # print("input_4th_1.shape", input_4th_1.shape)

        down_4_0 = self.down_4_0(input_4th_0)  # 8C
        down_4_1 = self.down_4_1(input_4th_1)
        # print("down_4_0.shape", down_4_0.shape)
        # print("down_4_1.shape", down_4_1.shape)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        # print("down_4_0m.shape", down_4_0m.shape)
        # print("down_4_1m.shape", down_4_1m.shape)

        inputBridge = torch.cat((down_4_0m, down_4_1m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)

        bridge = self.bridge(inputBridge)       # bz * 512 * 15 * 15

        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        skip_1 = (down_4_0 + down_4_1) / 2.0
        skip_2 = (down_3_0 + down_3_1) / 2.0
        skip_3 = (down_2_0 + down_2_1) / 2.0
        skip_4 = (down_1_0 + down_1_1) / 2.0

        bridge = self.ca(bridge) * bridge
        skip_1 = self.ca1(skip_1) * skip_1
        skip_2 = self.ca2(skip_2) * skip_2
        skip_3 = self.ca3(skip_3) * skip_3
        skip_4 = self.ca4(skip_4) * skip_4

        x = self.upLayer1(bridge, skip_1)
        x = self.upLayer2(x, skip_2)
        x = self.upLayer3(x, skip_3)
        x = self.upLayer4(x, skip_4)

        return x


class LSTM0(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)

    def forward(self, xt):
        """
        :param xt:      bz * 5(num_class) * 240 * 240
        :return:
            hide_1:    bz * ngf(32) * 240 * 240
            cell_1:    bz * ngf(32) * 240 * 240
        """
        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell_1 = torch.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1


class LSTM(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

    def forward(self, xt, cell_t_1, hide_t_1):
        """
        :param xt:          bz * (5+32) * 240 * 240
        :param hide_t_1:    bz * ngf(32) * 240 * 240
        :param cell_t_1:    bz * ngf(32) * 240 * 240
        :return:
        """
        gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
        gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
        oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
        hide_t = ot * torch.tanh(cell_t)            # bz * ngf(32) * 240 * 240

        return cell_t, hide_t


class LSTM_MMUnet(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, temporal=3):
        super(LSTM_MMUnet, self).__init__()
        self.temporal = temporal
        self.mmunet = MMUnet(input_nc, output_nc, ngf)
        self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
        self.lstm = LSTM(in_c=output_nc , ngf=ngf)

        self.mmout = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        :param x:  5D tensor    bz * temporal * 4 * 240 * 240
        :return:
        """
        output = []
        mm_output = []
        cell = None
        hide = None
        for t in range(self.temporal):
            # print('x.shape', x.shape)
            im_t = x[:, t, :, :, :].cuda()                # bz * 4 * 240 * 240
            # im_t = x[:, t, :, :, :]
            mm_last = self.mmunet(im_t)              # bz * 32 * 240 * 240
            out_t = self.mmout(mm_last)              # bz * 5 * 240 * 240
            mm_output.append(out_t)
            lstm_in = torch.cat((out_t, mm_last), dim=1)  # bz * 37 * 240 * 240

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)

            out_t = self.out(hide)
            output.append(out_t)

        return torch.stack(mm_output, dim=1), torch.stack(output, dim=1)


if __name__ == "__main__":
    batch_size = 2
    num_classes = 2
    ngf = 32

    net = LSTM_MMUnet(1, num_classes, ngf=ngf, temporal=3)
    print("total parameter:" + str(netSize(net)))   # 2860,3315
    MRI = torch.randn(batch_size, 3, 2, 256, 256)    # bz * temporal * modal * W * H

    mmout, predict = net(MRI)
    print(mmout.shape)
    print(predict.shape)  # (2, 3, 5, 64, 64)


