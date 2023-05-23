import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import datetime

# Download training data from open datasets.
# root is the path where the train/test data is stored,
# train specifies training or test dataset,
# download=True downloads the data from the internet if it’s not available at root.
# transform and target_transform specify the feature and label transformations
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# todo： ToTensor() 可以将 PIL image或者 ndarray转换成一个FloatTensor， 并且可以将图片的像素值归一化到 [0., 1.]范围内


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(training_data.data.size())  # torch.Size([60000, 28, 28])
print(training_data.data[0])  # torch.Size([60000, 28, 28])
print(training_data.data[0].size())  # torch.Size([28, 28])
print(test_data)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
print(len(training_data))  # 60000 一维数组的长度
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.savefig("pic/FashionMNIST%s.png" % datetime.datetime.now().day)
# plt.show()


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(".............................")
print(train_features)
print(train_labels)
print(f"Feature batch shape: {train_features.size()}") # Feature batch shape: torch.Size([64, 1, 28, 28])
print(f"Labels batch shape: {train_labels.size()}") # Labels batch shape: torch.Size([64])
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")