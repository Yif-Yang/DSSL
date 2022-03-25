import torchvision.transforms as T
# try:
#     from torchvision.transforms import GaussianBlur
# except ImportError:
#     from .gaussian_blur import GaussianBlur
#     T.GaussianBlur = GaussianBlur
# from RandAugment.augmentations import RandAugment
from .rand_aug_custom import RandAugment
from albumentations import RandomGridShuffle
import numpy as np
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_mean_std = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
from PIL import Image
import random

import moco.loader
import moco.builder
def GridShuffle(img, grid=4):  # [4, 8]
    return Image.fromarray(np.uint8(RandomGridShuffle(grid=(grid, grid), always_apply=True)(image=np.array(img))['image']))

class gf():
    def __init__(self, grid=4):
        self.grid = grid
    def __call__(self, img):
        if random.random() < 0.1:
            return GridShuffle(img, grid=self.grid)
        else:
            return img


class SimSiamTransform():
    def __init__(self, image_size, mean_std=cifar_mean_std, N=2, M=9, hard_N=8, hard_M=16, grid=4):

        self.ra = RandAugment(N, M)
        self.hard_ra = RandAugment(hard_N, hard_M)
        self.gf = gf(grid)
        self.transform = T.Compose([
            self.ra,
            T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        ])
        self.transform_hard = T.Compose([
            self.hard_ra,
            T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.5, 0.5, 0.5, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            self.gf,
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x11 = self.transform_hard(x1)
        x21 = self.transform_hard(x2)
        x1 = self.to_tensor(x1)
        x2 = self.to_tensor(x2)
        x11 = self.to_tensor(x11)
        x21 = self.to_tensor(x21)
        return x1, x2, x11, x21
    def to_img(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x11 = self.transform_hard(x1)
        x21 = self.transform_hard(x2)
        return x1, x2, x11, x21
if __name__ == '__main__':
    trans = SimSiamTransform(224)
    l1 = [12, 13, 14, 15, 16]
    l0 = [4, 5, 11]
    imgs_name = [f'WechatIMG{str(i)}' for i in l1]
    import os
    for img_name in imgs_name:
        out_path = f'/Users/yangyifan28/Documents/京东论文/test_twice/{img_name}'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        img = Image.open(f'/Users/yangyifan28/Documents/京东论文/{img_name}.jpeg').convert('RGB')
        img.save(os.path.join(out_path, img_name + '_original.jpg'))
        for i in range(100):
            img_name_now = f'{img_name}_{i}'
            a = SimSiamTransform(224)
            img_ret = a.to_img(img)
            img_ret[0].save(os.path.join(out_path, img_name_now+'0.jpeg'))
            img_ret[1].save(os.path.join(out_path, img_name_now+'1.jpeg'))
            img_ret[2].save(os.path.join(out_path, img_name_now+'2.jpeg'))
            img_ret[3].save(os.path.join(out_path, img_name_now + '3.jpeg'))
