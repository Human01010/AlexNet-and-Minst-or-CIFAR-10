import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import random

# 超参数
epochs = 10
batch_size = 128
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # AlexNet 需要输入 224x224 的图像
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像扩展为三通道
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三通道均值和标准差
])


transform_augment = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像扩展为三通道
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


# 加载 MNIST 数据集
train_dataset_org = datasets.MNIST(root="./data", train=True, transform=transform)
train_dataset_aug = datasets.MNIST(root="./data", train=True, transform=transform_augment)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

train_dataset = ConcatDataset([train_dataset_org, train_dataset_aug])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 加载预训练的 AlexNet 模型
model = models.alexnet(weights=None)
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 有 10 类

# 加载 CIFAR-10 的预训练权重
checkpoint = torch.load("cifarckpt.ckpt",weights_only=True)
model.load_state_dict(checkpoint)

model.classifier[6] = nn.Linear(4096, 10)  # MNIST 也有 10 类

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 冻结前面的卷积层，只训练最后一层卷积层和分类层
for param in model.features[:10].parameters():
    param.requires_grad = False

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


# 训练模型
def train():
    model.train()
    acc_list=[]
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # if (i + 1) % 100 == 0:  # 每 100 个 batch 打印一次日志
                # print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                # print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")

        acc = correct / total
        acc_list.append(acc)
        print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {100 * acc:.2f}%")
    return acc_list

# 测试模型
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    

if __name__ == "__main__":
    acc_list=train()
    test()
    plt.plot(acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on train dataset')
    plt.show()
