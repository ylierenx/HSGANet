import torch
import xlwt

import model_new
import copy
import cv2
import numpy as np
import time
import scipy.io as scio

from torchvision.utils import save_image
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from scipy.signal import convolve2d

block_size = 32
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")
load_flag = True
sr = 0.1
sampling_rate_1 = sr
sampling_rate = str(sr)+"_"
test_video_name = './dataset3'
test_video_output = test_video_name+'_output'

sample_and_inirecon = model_new.sample_and_inirecon(num_filters1=int(int(sampling_rate_1*1024)*0.8), num_filters2=int(sampling_rate_1*1024)-int(int(sampling_rate_1*1024)*0.8), B_size=32)
sample_and_inirecon.to(device)
fusion_ini = model_new.fusion_ini_9_small()
fusion_ini.to(device)

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

if load_flag:
    dict1 = torch.load('./final_ckpt/tvc_'+str(sampling_rate_1)+'_12phase_new.ckpt')
    sample_and_inirecon.load_state_dict(dict1['state_dict_sample_and_inirecon'])
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


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

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


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def psnr(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)
    max_gray = 1.
    mse = np.mean(np.power(img_rec - img_ori, 2))
    return 10. * np.log10(max_gray ** 2 / mse)


if __name__ == '__main__':
    workbook = xlwt.Workbook(encoding='utf-8')
    workbook_ssim = xlwt.Workbook(encoding='utf-8')


    print('test')

    for dataset_name in sorted(os.listdir(test_video_name)):
        print(dataset_name)
        file_names = []
        fnames = []
        psnr_value = []
        ssim_list = []
        dataset_n = os.path.join(test_video_name, dataset_name)
        worksheet = workbook.add_sheet(dataset_name)
        worksheet_ssim = workbook_ssim.add_sheet(dataset_name)

        if not os.path.exists(test_video_output):
            os.mkdir(test_video_output)
        if not os.path.exists(test_video_output + '/' + sampling_rate):
            os.mkdir(test_video_output + '/' + sampling_rate)
        if not os.path.exists(test_video_output + '/' + sampling_rate + '/' + dataset_name):
            os.mkdir(test_video_output + '/' + sampling_rate + '/' + dataset_name)

        for file_name in sorted(os.listdir(dataset_n)):
            fnames.append(os.path.join(dataset_n, file_name))
            file_names.append(file_name)
        for k in range(len(fnames)):
            worksheet.write(k, 0, file_names[k])
            worksheet_ssim.write(k, 0, file_names[k])

        for i in range(len(fnames)):

            Img = cv2.imread(fnames[i], 1)

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:, :, 0]

            Iorg_y = Iorg_y / 255.

            I = torch.from_numpy(Iorg_y)
            I = I.type(torch.FloatTensor)
            # I = I.to(device)

            # input_compose = Compose([ToTensor()])
            # I = input_compose(I)
            I = I.unsqueeze(0)
            I = I.unsqueeze(0)
            # I = I[np.newaxis, np.newaxis, :, :]
            # I = I / 255.
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
            inputs = inputs.cuda()
            I = I.cuda()

            p_name1 = './permutation_matrix/' + str(iw) + '.mat'
            p_name1_t = './permutation_matrix/' + str(iw) + '_t' + '.mat'
            p_name2 = './permutation_matrix/' + str(ih) + '.mat'
            p_name2_t = './permutation_matrix/' + str(ih) + '_t' + '.mat'

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

            with torch.no_grad():

                start = time.time()

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
                spl11_x = basic_Block_11(spl10_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2,
                                       rand_sw_p2_t)
                x_output = basic_Block_12(spl11_x, phi_x1, phi_x2, True, rand_sw_p1, rand_sw_p1_t, rand_sw_p2,
                                       rand_sw_p2_t)


                end = time.time()
                print(end - start)

                # save_image(x_output.cpu()[0][:, :ih, :iw], './fpgnet_output/'+sampling_rate+'/'+dataset_name + '/' + file_names[i] + '.png')
                x_output = x_output.cpu().numpy()

                x_output = np.clip(x_output, 0., 1.)
                I = I.cpu().numpy()
                recon_img = x_output[0, 0, :ih, :iw]
                ori_img = I[0, 0, :ih, :iw]
                p1 = psnr(recon_img, ori_img)

                recon_img = np.round(recon_img * 255)
                ori_img = np.round(ori_img * 255)
                recon_img = np.clip(recon_img, 0, 255)
                ori_img = np.clip(ori_img, 0, 255)

                ssim_value = compute_ssim(recon_img, ori_img)

                psnr_value.append(p1)
                ssim_list.append(ssim_value)

                # recon_img = recon_img * 255.
                # recon_img = np.clip(recon_img, 0, 255)
                Img_rec_yuv[:, :, 0] = recon_img
                im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                cv2.imwrite(
                    test_video_output + '/' + sampling_rate + '/' + dataset_name + '/' + file_names[i] + '.png',
                    im_rec_rgb)

                worksheet.write(i, 1, p1)
                worksheet_ssim.write(i, 1, ssim_value)
                print(file_names[i])
                print(p1)
        psnr_m = np.mean(psnr_value)
        ssim_m = np.mean(ssim_list)
        worksheet.write(len(fnames), 1, psnr_m)
        worksheet_ssim.write(len(fnames), 1, ssim_m)
    workbook.save(test_video_output + '/' + sampling_rate + '/psnr_' + sampling_rate + '.xls')
    workbook_ssim.save(test_video_output + '/' + sampling_rate + '/ssim_' + sampling_rate + '.xls')








