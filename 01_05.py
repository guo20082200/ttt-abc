"""
AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD
自动微分

参考：
https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import datetime
from torchvision.transforms import ToTensor, Lambda

# 1. 当训练神经网络的时候，最常用的算法就是反向传播back propagation，在这个算法中，参数(模型的权重weights)会根据对应参数的损失函数的梯度进行调整
# 2. 为了计算这些梯度，PyTorch内置了一个微分引擎：torch.autograd，它支持对任意的计算图自动求微分
# 3. 考虑最简单的单层的神经网络，输入x，参数w和b，和一些损失函数，定义的方式如下：

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b  # matmul：两个 Tensor相乘
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # 损失函数： binary_cross_entropy_with_logits

# 4. 在这个网络里面，w和b都是参数，需要优化。因此我们需要计算损失函数关于w和b这两个参数的梯度，为了计算梯度，我们设置这些张量requires_grad=True
# 可以在创建 tensor 的时候通过 requires_grad 设置，也可以通过 x.requires_grad_(True)方法来设置


# 作用的tensors上的一个函数（实际上是类Function的一个实例）可以构造计算图
# 该函数实例知道在前向传播中如何计算函数，并且也知道在反向传播中如何计算微分
# 反向传播中一个反向传播函数的引用存储在tensor的属性 grad_fn 中
# 跟多信息可以参考：https://pytorch.org/docs/stable/autograd.html#function

print(f"Gradient function for z = {z.grad_fn}")  # Gradient function for z = <AddBackward0 object at 0x0000026A2E6AB430>
print(f"Gradient function for loss = {loss.grad_fn}")
# Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x0000026A2E6AB430>


# 为了优化神经网络中的权重(weights)参数, 我们需要计算loss函数的梯度，
# 即：我们需要求∂loss/∂w和∂loss/∂b，在一些给定的x和y的情况下
# 为了求微分，我们可以调用 loss.backward() 然后查询 w.grad and b.grad 即可

loss.backward()
print(w.grad)
# tensor([[0.2988, 0.1858, 0.0059],
#         [0.2988, 0.1858, 0.0059],
#         [0.2988, 0.1858, 0.0059],
#         [0.2988, 0.1858, 0.0059],
#         [0.2988, 0.1858, 0.0059]])
print(b.grad)
# tensor([0.2988, 0.1858, 0.0059])

# 注意事项：
# 1. 我们只能获取计算图上叶子节点的梯度属性，这里要求叶子节点的requires_grad设置为True，计算图上其他节点的梯度不可用
# 2. 出于性能考虑，我们只能通过backward传播计算梯度一次，如果我们在一个计算图上调用backward多次，需要传递retain_graph=True来调用backward


# Disabling Gradient Tracking 禁止梯度追踪
# 默认情况下，requires_grad=True的所有的tensors可以追钟他们的计算历史，并且支持梯度的计算
# 然后，有些情况下我们不需要计算梯度，例如：当我们已经训练模型来作用到一些输入数据上的时候，
# 也就是说：我们仅仅需要的是正常传播，我们就停止追踪计算，通过 with torch.no_grad()代码块
z = torch.matmul(x, w) + b
print(z.requires_grad)  # True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)  # False

# 另一种方式获取相同的结果就是通过tensor的detach() 方法
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)  # False



# 关于计算图的更多知识
# 1. 概念上来说，在一个由函数对象组成的DAG里面，autograd包含了tensors并且和所有的已经执行的操作(伴随了新产生的tensor)，
# 叶子节点是输入的tensors，roots是输出的tensors，通过追踪计算图从roots到leaves，根据chain的规则你可以自动的计算出梯度
# 2. 在前向传播中，自动求导干了两件事情：
#   a. 运行需要的操作来计算出最终的结果 tensor
#   b. 在DAG里面维护了操作的梯度函数

# 3. 反向传播，在root.autograd形成的DAG上调用 backward()
#   a. 计算梯度，求出来每一个 .grad_fn
#   b. 分别在 tensor’s .grad属性里面累加
#   c. 使用链式规则(chain rule), 传播所有的数据到叶子节点张量(leaf tensors)


