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

class aug_combine():
    def __init__(self, N, M, grid=4):
        self.ra = RandAugment(N, M)
        self.grid = grid
    def __call__(self, img):
        if random.random() > 0.1:
            return self.ra(img)
        return GridShuffle(img, grid=self.grid)

class SimSiamTransform():
    def __init__(self, image_size, mean_std=cifar_mean_std, N=2, M=9, grid=4):

        self.aug_combine = aug_combine(N, M, grid)

        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        ])
        self.transform_rand_aug = T.Compose([
            self.aug_combine,
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x11 = self.transform_rand_aug(x1)
        x21 = self.transform_rand_aug(x2)
        x1 = self.to_tensor(x1)
        x2 = self.to_tensor(x2)
        x11 = self.to_tensor(x11)
        x21 = self.to_tensor(x21)
        return x1, x2, x11, x21