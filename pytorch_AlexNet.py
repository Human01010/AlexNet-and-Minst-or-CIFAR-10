import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt

epochs = 10
batch_size = 128
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # AlexNet 需要输入 224x224 的图像
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_augment = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),  
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
train_dataset_org = datasets.CIFAR10(root="./data2", train=True, transform=transform)
train_dataset_aug = datasets.CIFAR10(root="./data2", train=True, transform=transform_augment)
test_dataset = datasets.CIFAR10(root="./data2", train=False, transform=transform)

train_dataset = ConcatDataset([train_dataset_org, train_dataset_aug])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 加载 AlexNet 模型并修改分类层
model = models.alexnet(weights=None)
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 有 10 类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            #     # print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")


        acc = correct / total
        acc_list.append(acc)
        # print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * acc:.2f}%")
        print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {100 * acc:.2f}%")

    # 保存模型 checkpoint
    torch.save(model.state_dict(), "cifarckpt.ckpt")

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
    print(f"Test Accuracy: {100 * correct/total:.2f}%")


if __name__ == "__main__":
    acc_list=train()
    test()
    plt.plot(acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on train dataset')
    plt.savefig('cifar10.png')
    # plt.show()
