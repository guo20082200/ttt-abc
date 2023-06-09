import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
from torchvision.io import read_image


# 自定义dataset必须实现三个函数  __init__, __len__, and __getitem__
# 可以参考：FashionMNIST 的实现，图片存在img_dir， labels存在CSV里面
#
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)  # transform modify the features
        if self.target_transform:
            label = self.target_transform(label)  # target_transform modify the labels
        return image, label

