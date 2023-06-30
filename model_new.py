import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
import os
import sys
from torch.autograd import Variable

class softmax_gate_net_small(nn.Module):
    def __init__(self):
        super(softmax_gate_net_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        # x = F.relu(x, inplace=True)
        # x = self.conv5(x)
        output = F.softmax(x, dim=1)
        return output

class sample_and_inirecon(nn.Module):
    def __init__(self, num_filters1, num_filters2, B_size):
        super(sample_and_inirecon, self).__init__()
        self.sample1 = nn.Conv2d(in_channels=1, out_channels=num_filters1, kernel_size=B_size, stride=B_size, padding=0,
                                 bias=False)
        self.sample2 = nn.Conv2d(in_channels=1, out_channels=num_filters2, kernel_size=B_size, stride=B_size, padding=0,
                                 bias=False)

        self.B_size = B_size
        self.t2image = nn.PixelShuffle(B_size)
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2

    def forward(self, x_ini, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag):
        sample_w1 = self.sample1.weight
        sample_w1 = torch.reshape(sample_w1, (self.num_filters1, (self.B_size * self.B_size)))
        sample_w_t1 = sample_w1.t()
        sample_w_t1 = torch.unsqueeze(sample_w_t1, 2)
        t_mat1 = torch.unsqueeze(sample_w_t1, 3)

        sample_w2 = self.sample2.weight
        sample_w2 = torch.reshape(sample_w2, (self.num_filters2, (self.B_size * self.B_size)))
        sample_w_t2 = sample_w2.t()
        sample_w_t2 = torch.unsqueeze(sample_w_t2, 2)
        t_mat2 = torch.unsqueeze(sample_w_t2, 3)
        if flag:
            x1 = torch.matmul(x_ini, rand_sw_p1)
            x1 = torch.matmul(rand_sw_p2, x1)
            x1 = self.sample1(x1)
            x1 = y1 - x1
            x1 = F.conv2d(x1, t_mat1, stride=1, padding=0)
            x1 = self.t2image(x1)
            x1 = torch.matmul(rand_sw_p2_t, x1)
            x1 = torch.matmul(x1, rand_sw_p1_t)

            x2 = self.sample2(x_ini)
            x2 = y2 - x2
            x2 = F.conv2d(x2, t_mat2, stride=1, padding=0)
            x2 = self.t2image(x2)
            return x1, x2
        else:
            x1 = torch.matmul(x_ini, rand_sw_p1)
            x1 = torch.matmul(rand_sw_p2, x1)
            phi_x1 = self.sample1(x1)
            x1 = F.conv2d(phi_x1, t_mat1, stride=1, padding=0)
            x1 = self.t2image(x1)
            x1 = torch.matmul(rand_sw_p2_t, x1)
            x1 = torch.matmul(x1, rand_sw_p1_t)

            phi_x2 = self.sample2(x_ini)
            x2 = F.conv2d(phi_x2, t_mat2, stride=1, padding=0)
            x2 = self.t2image(x2)
            return x1, phi_x1, x2, phi_x2

class Biv_Shr_small(nn.Module):
    def __init__(self):
        super(Biv_Shr_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_ini):
        # x_ini = x
        x = self.conv1(x_ini)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = x_ini + x

        return x


class fusion_ini_8_small(nn.Module):
    def __init__(self, in_filters):
        super(fusion_ini_8_small, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_filters, out_channels=32, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_ini):
        # x_ini = torch.cat((x1, x2), dim=1)
        x1 = self.conv1_1(x_ini)
        x1 = F.relu(x1, inplace=True)
        x1 = self.conv1_2(x1)
        x2 = self.conv2(x_ini)
        x3 = self.conv3(x_ini)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv4(x)
        # x = F.relu(x, inplace=True)
        return x


class fusion_ini_9_small(nn.Module):
    def __init__(self):
        super(fusion_ini_9_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.basic_fusion1 = fusion_ini_8_small(in_filters=32)
        self.basic_fusion2 = fusion_ini_8_small(in_filters=32)
        self.basic_fusion3 = fusion_ini_8_small(in_filters=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion1(x_in)
        x = x + x_in
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion2(x_in)
        x = x + x_in
        x_in = F.relu(x, inplace=True)
        x = self.basic_fusion3(x_in)
        x = x + x_in
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        return x














