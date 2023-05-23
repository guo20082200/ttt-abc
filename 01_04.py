"""
构建模型
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import datetime
from torchvision.transforms import ToTensor, Lambda

print(torch.backends.mps.is_available())  # False

# 获取设备类型
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")  # Using cuda device


# 1. Neural networks由执行操作数据的layers/modules组成，
# 2. torch.nn的命名空间提供了所有的构建模块来构建自己的神经网络
# 3. 每一个模块都是nn.Module的子类
# 4. 一个神经网络本身就是一个包含了其他模块（或layers）的模块
# 5. 这种嵌套的结构更加容易的适合构建和管理复杂的架构


# 1. 通过继承nn.Module来定义自己的neural network
# 2. 在__init__方法中初始化神经网络
# 3. 每一个nn.Module的子类都需要实现forward方法(用来操作input数据的方法)
# 4. 这里构建的神经网络用来做FashionMNIST数据集的图片分类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 创建NeuralNetwork的实例，放到device上计算，然后打印其结构
model = NeuralNetwork().to(device)
print(model)
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )

# 为了使用这个模型，传递数据给参数，，该模型会执行forward方法（背后还有一些计算），不需要直接调用model.forward()


X = torch.rand(3, 28, 28, device=device)
print(X)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)  # 使用 Softmax做为损失函数
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")  # Predicted class: tensor([6, 6, 6], device='cuda:0')

input_image = torch.rand(3, 28, 28)
print(input_image.size())  # torch.Size([3, 28, 28])

# 初始化 nn.Flatten， 转换一个28x28的图片为一个连续的784像素的数组(最小批的dim=0，保持)
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())  # torch.Size([3, 784])

# nn.Linear模块，用自身存储的权重和biases 来做了一个线性变换
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())  # torch.Size([3, 20])


# nn.ReLU()
# 非线性的激活函数：在输入和输出之间创建复杂的映射，应用在线性变换之后引入非线性，帮助神经网络来学习一个更大范围的场景
# 在这个模型中，我们在线性曾之间使用nn.ReLU()，但是模型中还有其他的非线性的激活函数

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Before ReLU: tensor([[-0.2941, -0.0165, -0.8509,  0.0085,  0.0823,  0.0605,  0.0151, -0.2411,
#           0.2212, -0.0643,  0.6488,  0.1092,  0.4309,  0.5476,  0.1635,  0.0145,
#          -0.0792,  0.1919,  0.0920, -0.1995],
#         [-0.0836,  0.3009, -1.1770, -0.3256, -0.0037,  0.1428,  0.2413, -0.1651,
#           0.0726, -0.2595,  0.6902,  0.1873,  0.1849,  0.4229,  0.2533, -0.2491,
#          -0.4350, -0.0455,  0.1705,  0.0998],
#         [-0.4492,  0.0926, -1.4539,  0.1471,  0.3369,  0.1907,  0.5566, -0.2043,
#           0.2821, -0.2944,  0.3912, -0.0737, -0.0635,  0.7752,  0.5784,  0.0164,
#          -0.3973,  0.3215,  0.4347, -0.1387]], grad_fn=<AddmmBackward0>)
#
#
# After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.0085, 0.0823, 0.0605, 0.0151, 0.0000, 0.2212,
#          0.0000, 0.6488, 0.1092, 0.4309, 0.5476, 0.1635, 0.0145, 0.0000, 0.1919,
#          0.0920, 0.0000],
#         [0.0000, 0.3009, 0.0000, 0.0000, 0.0000, 0.1428, 0.2413, 0.0000, 0.0726,
#          0.0000, 0.6902, 0.1873, 0.1849, 0.4229, 0.2533, 0.0000, 0.0000, 0.0000,
#          0.1705, 0.0998],
#         [0.0000, 0.0926, 0.0000, 0.1471, 0.3369, 0.1907, 0.5566, 0.0000, 0.2821,
#          0.0000, 0.3912, 0.0000, 0.0000, 0.7752, 0.5784, 0.0164, 0.0000, 0.3215,
#          0.4347, 0.0000]], grad_fn=<ReluBackward0>)


# nn.Sequential
# nn.Sequential 包含了很多模块的有序容器.
# 数据按照指定好的顺序穿过所有的模块
# 你可以使用 sequential 容器放在一起，就像seq_modules
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print(logits)
# tensor([[-0.1673, -0.1190, -0.1801,  0.0567, -0.0803, -0.2515,  0.0584, -0.0985,
#          -0.0114, -0.1051],
#         [-0.1270, -0.0939, -0.2313,  0.2873,  0.0069, -0.4677, -0.0316, -0.0940,
#          -0.1751,  0.0817],
#         [-0.1843, -0.1494, -0.2028,  0.1651, -0.0541, -0.2616,  0.1360, -0.0723,
#          -0.1054, -0.1511]], grad_fn=<AddmmBackward0>)


# nn.Softmax
# 神经网络的最后一层返回的数据是：logits，原始数据的范围是： [-infty, infty]，这个数据传递给了nn.Softmax模块
# nn.Softmax模块把数据scale 到了 [0, 1] 之间，代表了该模型针对每一个分类的预测概率
# dim参数表示该维度的数据求和结果必须是1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)
# tensor([[0.0827, 0.0962, 0.0957, 0.1046, 0.0796, 0.0801, 0.1120, 0.1162, 0.0889,
#          0.1440],
#         [0.0777, 0.0989, 0.0984, 0.1212, 0.0666, 0.0804, 0.1110, 0.1033, 0.0763,
#          0.1662],
#         [0.0787, 0.1007, 0.0929, 0.1223, 0.0710, 0.0803, 0.1080, 0.1118, 0.0772,
#          0.1572]], grad_fn=<SoftmaxBackward0>)


# 模型的参数
# 神经网络内部的多层都是参数化的，也就是说有与其相关联的weights 和 biases，这两部分数据在训练的时候会被优化
# nn.Module的子类会自动的追踪用户自定义的模型对象里面的所有字段
# 并且 使得所有的参数都是可以访问的，通过模型的parameters()方法 或者 named_parameters()方法
print(f"Model structure: {model}\n\n")

# 这里我们迭代每一个参数，打印出来参数的大小并预览参数里面包含的值
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#
# Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0257, -0.0251,  0.0081,  ...,  0.0178,  0.0334,  0.0225],
#         [ 0.0339, -0.0110,  0.0333,  ...,  0.0316, -0.0272,  0.0354]],
#        device='cuda:0', grad_fn=<SliceBackward0>)
#
# Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([ 0.0122, -0.0179], device='cuda:0', grad_fn=<SliceBackward0>)
#
# Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0231, -0.0005,  0.0091,  ...,  0.0289,  0.0132, -0.0143],
#         [-0.0353,  0.0268,  0.0062,  ...,  0.0370,  0.0197, -0.0030]],
#        device='cuda:0', grad_fn=<SliceBackward0>)
#
# Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0359, 0.0128], device='cuda:0', grad_fn=<SliceBackward0>)
#
# Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0090,  0.0061, -0.0122,  ...,  0.0214, -0.0272, -0.0113],
#         [ 0.0259, -0.0002,  0.0122,  ...,  0.0137, -0.0194,  0.0292]],
#        device='cuda:0', grad_fn=<SliceBackward0>)
#
# Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0283, 0.0155], device='cuda:0', grad_fn=<SliceBackward0>)
