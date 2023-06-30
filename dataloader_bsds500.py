from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip
from PIL import Image
import numpy as np
import os
import sys


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class bsds_500(Dataset):
    def __init__(self, image_height=256, image_width=256, load_filename='bsds500_train.npy'):
        # self.output_dir = '../dataset'
        # self.fnames = []
        # for label in sorted(os.listdir(self.output_dir + '/train_test')):
        #     self.fnames.append(os.path.join(self.output_dir + '/train_test', label))
        # np.save('bsds500_train.npy', self.fnames)
        # self.fnames = []
        # for label in sorted(os.listdir(self.output_dir + '/val')):
        #     self.fnames.append(os.path.join(self.output_dir + '/val', label))
        # np.save('bsds500_val.npy', self.fnames)
        # sys.exit()
        self.load_filename = load_filename
        self.fnames = np.load(self.load_filename)

        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        image = load_img(self.fnames[idx])
        input_compose = Compose([RandomCrop(self.image_height), RandomHorizontalFlip(),
                                 RandomVerticalFlip(), ToTensor()])
        image = input_compose(image)
        return image
# if __name__=='__main__':
#     mybsds500 = bsds_500()

