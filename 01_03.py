import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import datetime
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# transform： ToTensor() 可以将 PIL image或者 ndarray转换成一个FloatTensor， 并且可以将图片的像素值归一化到 [0., 1.]范围内
# target_transform: Lambda 转换器可以适用于任何的用户自定义的lambda函数
# 这里定义了一个函数来将integer转换为一个one-hot encoded tensor
# 第一步创建了一个size是10（我们这里的dataset中的labels的数量）的zero tensor
# 然后调用了scatter_函数，对于给定的标签y，这里分配一个value=1
