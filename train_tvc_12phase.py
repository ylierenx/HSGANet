import scipy.io as scio
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model_new
import dataloader_bsds500
from math import log10
import sys
from PIL import Image
import numpy as np
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
from torchvision.utils import save_image
import matplotlib.image as mpimg
import os
from torchvision.transforms import Compose, ToTensor

block_size = 32
load_flag = False
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0")
sr = 0.1
ratio = 0.8
sample_and_inirecon = model_new.sample_and_inirecon(num_filters1=int(1024*sr*ratio), num_filters2=int(1024*sr)-int(1024*sr*ratio), B_size=32)
sample_and_inirecon.to(device)
fusion_ini = model_new.fusion_ini_9_small()
fusion_ini.to(device)

checkpoint_counter = 1

Biv_Shr0 = model_new.Biv_Shr_small()
Biv_Shr0.to(device)
Biv_Shr1 = model_new.Biv_Shr_small()
Biv_Shr1.to(device)
Biv_Shr2 = model_new.Biv_Shr_small()
Biv_Shr2.to(device)
Biv_Shr3 = model_new.Biv_Shr_small()
Biv_Shr3.to(device)
Biv_Shr4 = model_new.Biv_Shr_small()
Biv_Shr4.to(device)
Biv_Shr5 = model_new.Biv_Shr_small()
Biv_Shr5.to(device)
Biv_Shr6 = model_new.Biv_Shr_small()
Biv_Shr6.to(device)
Biv_Shr7 = model_new.Biv_Shr_small()
Biv_Shr7.to(device)
Biv_Shr8 = model_new.Biv_Shr_small()
Biv_Shr8.to(device)
Biv_Shr9 = model_new.Biv_Shr_small()
Biv_Shr9.to(device)
Biv_Shr10 = model_new.Biv_Shr_small()
Biv_Shr10.to(device)
Biv_Shr11 = model_new.Biv_Shr_small()
Biv_Shr11.to(device)
Biv_Shr12 = model_new.Biv_Shr_small()
Biv_Shr12.to(device)
softmax_gate_net1 = model_new.softmax_gate_net_small()
softmax_gate_net1.to(device)
softmax_gate_net2 = model_new.softmax_gate_net_small()
softmax_gate_net2.to(device)
softmax_gate_net3 = model_new.softmax_gate_net_small()
softmax_gate_net3.to(device)
softmax_gate_net4 = model_new.softmax_gate_net_small()
softmax_gate_net4.to(device)
softmax_gate_net5 = model_new.softmax_gate_net_small()
softmax_gate_net5.to(device)
softmax_gate_net6 = model_new.softmax_gate_net_small()
softmax_gate_net6.to(device)
softmax_gate_net7 = model_new.softmax_gate_net_small()
softmax_gate_net7.to(device)
softmax_gate_net8 = model_new.softmax_gate_net_small()
softmax_gate_net8.to(device)
softmax_gate_net9 = model_new.softmax_gate_net_small()
softmax_gate_net9.to(device)
softmax_gate_net10 = model_new.softmax_gate_net_small()
softmax_gate_net10.to(device)
softmax_gate_net11 = model_new.softmax_gate_net_small()
softmax_gate_net11.to(device)
softmax_gate_net12 = model_new.softmax_gate_net_small()
softmax_gate_net12.to(device)

params_softmax_gate_net = list(softmax_gate_net1.parameters()) + list(softmax_gate_net2.parameters()) + list(
    softmax_gate_net3.parameters()) + list(softmax_gate_net4.parameters()) + list(
    softmax_gate_net5.parameters()) + list(softmax_gate_net6.parameters()) + list(
    softmax_gate_net7.parameters()) + list(softmax_gate_net8.parameters()) + list(
    softmax_gate_net9.parameters()) + list(softmax_gate_net10.parameters()) + list(
    softmax_gate_net11.parameters()) + list(softmax_gate_net12.parameters())

params_fusion_ini = list(fusion_ini.parameters())

params_ininet = list(sample_and_inirecon.parameters())

params_Biv_Shr = list(Biv_Shr0.parameters()) + list(Biv_Shr1.parameters()) + list(Biv_Shr2.parameters()) + list(Biv_Shr3.parameters()) + list(
    Biv_Shr4.parameters()) + list(Biv_Shr5.parameters()) + list(Biv_Shr6.parameters()) + list(Biv_Shr7.parameters()) + list(Biv_Shr8.parameters()) + list(
    Biv_Shr9.parameters()) + list(Biv_Shr10.parameters()) + list(Biv_Shr11.parameters()) + list(Biv_Shr12.parameters())


params_splnet_fusion = params_Biv_Shr + params_ininet + params_fusion_ini + params_softmax_gate_net

optimizer_splnet_fusion = optim.Adam(params_splnet_fusion, lr=0.0001)

if load_flag:
    dict1 = torch.load('./check_point1/tci_0.25_12phase_new_4000.ckpt')
    sample_and_inirecon.load_state_dict(dict1['state_dict_sample_and_inirecon'])
    optimizer_splnet_fusion.load_state_dict(dict1['state_dict_optimizer_splnet_fusion'])
    fusion_ini.load_state_dict(dict1['state_dict_fusion_ini'])
    Biv_Shr0.load_state_dict(dict1['state_dict_Biv_Shr0'])
    Biv_Shr1.load_state_dict(dict1['state_dict_Biv_Shr1'])
    Biv_Shr2.load_state_dict(dict1['state_dict_Biv_Shr2'])
    Biv_Shr3.load_state_dict(dict1['state_dict_Biv_Shr3'])
    Biv_Shr4.load_state_dict(dict1['state_dict_Biv_Shr4'])
    Biv_Shr5.load_state_dict(dict1['state_dict_Biv_Shr5'])
    Biv_Shr6.load_state_dict(dict1['state_dict_Biv_Shr6'])
    Biv_Shr7.load_state_dict(dict1['state_dict_Biv_Shr7'])
    Biv_Shr8.load_state_dict(dict1['state_dict_Biv_Shr8'])
    Biv_Shr9.load_state_dict(dict1['state_dict_Biv_Shr9'])
    Biv_Shr10.load_state_dict(dict1['state_dict_Biv_Shr10'])
    Biv_Shr11.load_state_dict(dict1['state_dict_Biv_Shr11'])
    Biv_Shr12.load_state_dict(dict1['state_dict_Biv_Shr12'])
    softmax_gate_net1.load_state_dict(dict1['state_dict_softmax_gate_net1'])
    softmax_gate_net2.load_state_dict(dict1['state_dict_softmax_gate_net2'])
    softmax_gate_net3.load_state_dict(dict1['state_dict_softmax_gate_net3'])
    softmax_gate_net4.load_state_dict(dict1['state_dict_softmax_gate_net4'])
    softmax_gate_net5.load_state_dict(dict1['state_dict_softmax_gate_net5'])
    softmax_gate_net6.load_state_dict(dict1['state_dict_softmax_gate_net6'])
    softmax_gate_net7.load_state_dict(dict1['state_dict_softmax_gate_net7'])
    softmax_gate_net8.load_state_dict(dict1['state_dict_softmax_gate_net8'])
    softmax_gate_net9.load_state_dict(dict1['state_dict_softmax_gate_net9'])
    softmax_gate_net10.load_state_dict(dict1['state_dict_softmax_gate_net10'])
    softmax_gate_net11.load_state_dict(dict1['state_dict_softmax_gate_net11'])
    softmax_gate_net12.load_state_dict(dict1['state_dict_softmax_gate_net12'])


trainset = dataloader_bsds500.bsds_500(image_height=96, image_width=96, load_filename='bsds500_train.npy')
train_loader = dataloader_bsds500.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True,
                                             drop_last=True)

def psnr(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)

    max_gray = 1.
    mse = np.mean(np.power(img_rec - img_ori, 2))

    return 10. * np.log10(max_gray ** 2 / mse)


def basic_Block_1(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net1(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr1(x)
    return x

def basic_Block_2(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net2(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr2(x)
    return x

def basic_Block_3(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net3(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr3(x)
    return x

def basic_Block_4(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net4(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr4(x)
    return x

def basic_Block_5(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net5(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr5(x)
    return x

def basic_Block_6(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net6(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr6(x)
    return x

def basic_Block_7(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net7(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr7(x)
    return x

def basic_Block_8(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net8(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr8(x)
    return x

def basic_Block_9(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net9(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr9(x)
    return x

def basic_Block_10(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net10(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr10(x)
    return x

def basic_Block_11(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net11(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr11(x)
    return x

def basic_Block_12(x, y1, y2, flag, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t):
    x1, x2 = sample_and_inirecon(x, y1, y2, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t, flag)
    prob = softmax_gate_net12(x1, x2)
    x = x + prob[:, [0], :, :] * x1 + prob[:, [1], :, :] * x2
    x = Biv_Shr12(x)
    return x

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test_hsganet(device):
    for dataset_name in sorted(os.listdir('./dataset1')):
        psnr_value = []
        fnames = []
        dataset_n = os.path.join('./dataset1', dataset_name)
        for file_name in sorted(os.listdir(dataset_n)):
            fnames.append(os.path.join(dataset_n, file_name))

        with torch.no_grad():
            for f_i in range(len(fnames)):
                img = Image.open(fnames[f_i])
                I = np.array(img)

                I = Image.fromarray(I)

                input_compose = Compose([ToTensor()])
                I = input_compose(I)
                I = I.unsqueeze(0)

                inputs = I

                ih = I.shape[2]
                iw = I.shape[3]

                if np.mod(iw, block_size) != 0:
                    col_pad = block_size - np.mod(iw, block_size)
                    inputs = torch.cat((inputs, torch.zeros([1, 1, ih, col_pad])), axis=3)
                else:
                    col_pad = 0
                    inputs = inputs
                if np.mod(ih, block_size) != 0:
                    row_pad = block_size - np.mod(ih, block_size)
                    inputs = torch.cat((inputs, torch.zeros([1, 1, row_pad, iw + col_pad])), axis=2)
                else:
                    row_pad = 0

                ih_1 = inputs.shape[2]
                iw_1 = inputs.shape[3]
                p_name1 = './permutation_matrix/' + str(iw_1) + '.mat'
                p_name1_t = './permutation_matrix/' + str(iw_1) + '_t' + '.mat'
                p_name2 = './permutation_matrix/' + str(ih_1) + '.mat'
                p_name2_t = './permutation_matrix/' + str(ih_1) + '_t' + '.mat'

                # print(p_name1)
                # sys.exit()

                data = scio.loadmat(p_name1)
                rand_sw_p1 = data['rand_sw_p1']
                rand_sw_p1 = np.array(rand_sw_p1).astype(np.float32)
                data_t = scio.loadmat(p_name1_t)
                rand_sw_p1_t = data_t['rand_sw_p1_t']
                rand_sw_p1_t = np.array(rand_sw_p1_t).astype(np.float32)

                data = scio.loadmat(p_name2)
                rand_sw_p2 = data['rand_sw_p1']
                rand_sw_p2 = np.array(rand_sw_p2).astype(np.float32)
                data_t = scio.loadmat(p_name2_t)
                rand_sw_p2_t = data_t['rand_sw_p1_t']
                rand_sw_p2_t = np.array(rand_sw_p2_t).astype(np.float32)

                rand_sw_p1 = torch.from_numpy(rand_sw_p1)
                rand_sw_p1_t = torch.from_numpy(rand_sw_p1_t)
                rand_sw_p1 = rand_sw_p1.to(device)
                rand_sw_p1_t = rand_sw_p1_t.to(device)

                rand_sw_p2 = torch.from_numpy(rand_sw_p2)
                rand_sw_p2_t = torch.from_numpy(rand_sw_p2_t)
                rand_sw_p2 = rand_sw_p2.to(device)
                rand_sw_p2_t = rand_sw_p2_t.to(device)

                inputs = inputs.to(device)

                ini_x1, phi_x1, ini_x2, phi_x2 = sample_and_inirecon(inputs, 0, 0, rand_sw_p1, rand_sw_p1_t, rand_sw_p2,
                                                                     rand_sw_p2_t, False)

                ini_x = fusion_ini(ini_x1, ini_x2)
                ini_x = Biv_Shr0(ini_x)

                spl1_x = basic_Block_1(ini_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl2_x = basic_Block_2(spl1_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl3_x = basic_Block_3(spl2_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl4_x = basic_Block_4(spl3_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl5_x = basic_Block_5(spl4_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl6_x = basic_Block_6(spl5_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl7_x = basic_Block_7(spl6_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl8_x = basic_Block_8(spl7_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl9_x = basic_Block_9(spl8_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl10_x = basic_Block_10(spl9_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl11_x = basic_Block_11(spl10_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)
                spl12_x = basic_Block_12(spl11_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2, rand_sw_p2_t)


                x_output = spl12_x.cpu().numpy()
                I = I.cpu().numpy()
                recon_img = x_output[0, 0, :ih, :iw]
                ori_img = I[0, 0, :ih, :iw]
                p1 = psnr(recon_img, ori_img)
                psnr_value.append(p1)

    return np.mean(psnr_value)


if __name__ == '__main__':

    data = scio.loadmat('permutation_matrix/96.mat')
    rand_sw_p1 = data['rand_sw_p1']
    rand_sw_p1 = np.array(rand_sw_p1).astype(np.float32)

    data_t = scio.loadmat('permutation_matrix/96_t.mat')
    rand_sw_p1_t = data_t['rand_sw_p1_t']
    rand_sw_p1_t = np.array(rand_sw_p1_t).astype(np.float32)

    rand_sw_p1 = torch.from_numpy(rand_sw_p1)
    rand_sw_p1_t = torch.from_numpy(rand_sw_p1_t)

    rand_sw_p1_train = rand_sw_p1.to(device)
    rand_sw_p1_t_train = rand_sw_p1_t.to(device)

    start = time.time()

    for epoch in range(0, 26001):

        if (epoch >= 25000) and (epoch < 25500):
            lr = 0.00001
            adjust_learning_rate(optimizer_splnet_fusion, lr)
        if epoch >= 25500:
            lr = 0.000001
            adjust_learning_rate(optimizer_splnet_fusion, lr)
        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device)

            optimizer_splnet_fusion.zero_grad()

            ini_x1, phi_x1, ini_x2, phi_x2 = sample_and_inirecon(inputs, 0, 0, rand_sw_p1_train, rand_sw_p1_t_train,
                                                                 rand_sw_p1_train,
                                                                 rand_sw_p1_t_train, False)

            ini_x = fusion_ini(ini_x1, ini_x2)
            ini_x = Biv_Shr0(ini_x)

            spl1_x = basic_Block_1(ini_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl2_x = basic_Block_2(spl1_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl3_x = basic_Block_3(spl2_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl4_x = basic_Block_4(spl3_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl5_x = basic_Block_5(spl4_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl6_x = basic_Block_6(spl5_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl7_x = basic_Block_7(spl6_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl8_x = basic_Block_8(spl7_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl9_x = basic_Block_9(spl8_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl10_x = basic_Block_10(spl9_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl11_x = basic_Block_11(spl10_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)
            spl12_x = basic_Block_12(spl11_x, phi_x1, phi_x2, True, rand_sw_p1_train, rand_sw_p1_t_train, rand_sw_p1_train,
                                 rand_sw_p1_t_train)


            recnLoss_final = torch.mean(
                torch.norm((inputs - spl12_x), p=2, dim=(2, 3)) * torch.norm((inputs - spl12_x), p=2, dim=(2, 3)))

            recnLoss_all = recnLoss_final

            recnLoss_all.backward()
            optimizer_splnet_fusion.step()

            if ((i % 6) == 0 and epoch % 8 == 0):
                print('test')
                p1 = test_hsganet(device)

                print(p1)
                print("final_loss: %0.6f " % (
                    recnLoss_final.item()))
                print("train_loss: %0.6f  Iterations: %4d/%4d epoch:%d " % (
                    recnLoss_all.item(), i, len(train_loader), epoch))

                f = open('test_tvc_'+str(sr)+'_12phase.txt', 'a')
                f.write(" %0.6f %d " % (p1, epoch))
                f.write('\n')
                f.close()
                end = time.time()
                print(end - start)
                start = time.time()
        if (epoch < 25000 and epoch%1000==0 and epoch>0):

            dict1 = {

                'epoch': epoch,

                'state_dict_sample_and_inirecon': sample_and_inirecon.state_dict(),
                'state_dict_Biv_Shr0': Biv_Shr0.state_dict(),
                'state_dict_Biv_Shr1': Biv_Shr1.state_dict(),
                'state_dict_Biv_Shr2': Biv_Shr2.state_dict(),
                'state_dict_Biv_Shr3': Biv_Shr3.state_dict(),
                'state_dict_Biv_Shr4': Biv_Shr4.state_dict(),
                'state_dict_Biv_Shr5': Biv_Shr5.state_dict(),
                'state_dict_Biv_Shr6': Biv_Shr6.state_dict(),
                'state_dict_Biv_Shr7': Biv_Shr7.state_dict(),
                'state_dict_Biv_Shr8': Biv_Shr8.state_dict(),
                'state_dict_Biv_Shr9': Biv_Shr9.state_dict(),
                'state_dict_Biv_Shr10': Biv_Shr10.state_dict(),
                'state_dict_Biv_Shr11': Biv_Shr11.state_dict(),
                'state_dict_Biv_Shr12': Biv_Shr12.state_dict(),
                'state_dict_optimizer_splnet_fusion': optimizer_splnet_fusion.state_dict(),
                'state_dict_softmax_gate_net1': softmax_gate_net1.state_dict(),
                'state_dict_softmax_gate_net2': softmax_gate_net2.state_dict(),
                'state_dict_softmax_gate_net3': softmax_gate_net3.state_dict(),
                'state_dict_softmax_gate_net4': softmax_gate_net4.state_dict(),
                'state_dict_softmax_gate_net5': softmax_gate_net5.state_dict(),
                'state_dict_softmax_gate_net6': softmax_gate_net6.state_dict(),
                'state_dict_softmax_gate_net7': softmax_gate_net7.state_dict(),
                'state_dict_softmax_gate_net8': softmax_gate_net8.state_dict(),
                'state_dict_softmax_gate_net9': softmax_gate_net9.state_dict(),
                'state_dict_softmax_gate_net10': softmax_gate_net10.state_dict(),
                'state_dict_softmax_gate_net11': softmax_gate_net11.state_dict(),
                'state_dict_softmax_gate_net12': softmax_gate_net12.state_dict(),
                'state_dict_fusion_ini': fusion_ini.state_dict()

            }
            torch.save(dict1, './check_point' + '/tvc_' + str(sr) + '_12phase_new_' + str(epoch) + ".ckpt")
        if (epoch > 25000 and  epoch<26001 and epoch%500==0):

            dict1 = {

                'epoch': epoch,

                'state_dict_sample_and_inirecon': sample_and_inirecon.state_dict(),
                'state_dict_Biv_Shr0': Biv_Shr0.state_dict(),
                'state_dict_Biv_Shr1': Biv_Shr1.state_dict(),
                'state_dict_Biv_Shr2': Biv_Shr2.state_dict(),
                'state_dict_Biv_Shr3': Biv_Shr3.state_dict(),
                'state_dict_Biv_Shr4': Biv_Shr4.state_dict(),
                'state_dict_Biv_Shr5': Biv_Shr5.state_dict(),
                'state_dict_Biv_Shr6': Biv_Shr6.state_dict(),
                'state_dict_Biv_Shr7': Biv_Shr7.state_dict(),
                'state_dict_Biv_Shr8': Biv_Shr8.state_dict(),
                'state_dict_Biv_Shr9': Biv_Shr9.state_dict(),
                'state_dict_Biv_Shr10': Biv_Shr10.state_dict(),
                'state_dict_Biv_Shr11': Biv_Shr11.state_dict(),
                'state_dict_Biv_Shr12': Biv_Shr12.state_dict(),
                'state_dict_optimizer_splnet_fusion': optimizer_splnet_fusion.state_dict(),
                'state_dict_softmax_gate_net1': softmax_gate_net1.state_dict(),
                'state_dict_softmax_gate_net2': softmax_gate_net2.state_dict(),
                'state_dict_softmax_gate_net3': softmax_gate_net3.state_dict(),
                'state_dict_softmax_gate_net4': softmax_gate_net4.state_dict(),
                'state_dict_softmax_gate_net5': softmax_gate_net5.state_dict(),
                'state_dict_softmax_gate_net6': softmax_gate_net6.state_dict(),
                'state_dict_softmax_gate_net7': softmax_gate_net7.state_dict(),
                'state_dict_softmax_gate_net8': softmax_gate_net8.state_dict(),
                'state_dict_softmax_gate_net9': softmax_gate_net9.state_dict(),
                'state_dict_softmax_gate_net10': softmax_gate_net10.state_dict(),
                'state_dict_softmax_gate_net11': softmax_gate_net11.state_dict(),
                'state_dict_softmax_gate_net12': softmax_gate_net12.state_dict(),
                'state_dict_fusion_ini': fusion_ini.state_dict()

            }
            torch.save(dict1, './check_point' + '/tvc_' + str(sr) + '_12phase_new_' + str(epoch) + ".ckpt")
