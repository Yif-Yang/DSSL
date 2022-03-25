
from torchvision.datasets.vision import VisionDataset
import os
import pickle
from torchvision.datasets.folder import default_loader

class Imagenet(VisionDataset):
    def __init__(self, root, data_list, train=True, transform=None, target_transform=None, img_dir='all', target_dir='annos'):

        super(Imagenet, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.data = []
        self.targets = []
        self.train = train
        self.data_list = os.path.join(root, data_list)
        self.img_dir_path = os.path.join(root, img_dir)
        self.target_dir_path = os.path.join(root, target_dir)
        self.transform = transform
        self.target_transform = target_transform
        if (os.path.isfile(self.data_list)):
            with open(self.data_list, 'r') as infile:
                for line in infile:
                    img_name, label = line.strip().split(' ')
                    self.data.append(os.path.join(self.img_dir_path, img_name))
                    self.targets.append(int(label) - 1)
        else:
            print('data list is not file')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = default_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
