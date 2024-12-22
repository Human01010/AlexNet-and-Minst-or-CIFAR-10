import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import datasets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformations for the MNIST dataset and the augmented MNIST dataset, combine them
transform_mnist = transforms.Compose([
    # transforms.Resize(96),  # Resize to 96x96 to match CIFAR-10 image size
    transforms.Resize(224),
transforms.Grayscale(num_output_channels=3),  # 将灰度图像扩展为三通道
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
])
transform_augment_mnist = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像扩展为三通道
    transforms.ToTensor(),
])

# Load the original dataset
original_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform_mnist, download=True)

# Load the augmented dataset
augmented_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform_augment_mnist, download=True)

# Combine the original and augmented datasets
combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

#train and validation dataset
train_set = combined_dataset
val_set = datasets.MNIST(root='dataset/', train=False, transform=transform_mnist, download=True)

# Create data loaders
train_loader = DataLoader(dataset=combined_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(dataset=datasets.MNIST(root='dataset/', train=False, transform=transform_mnist, download=True), batch_size=10, shuffle=False)

def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self._config = config
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self._config['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)

class Trainer(object):
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config['lr'], weight_decay=config['l2_regularization'])
        self.loss_func = nn.CrossEntropyLoss()
        self.train_accuracies = []
        self.test_accuracies = []

    def _train_single_batch(self, images, labels):
        y_predict = self._model(images)
        loss = self.loss_func(y_predict, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        loss = loss.item()
        _, predicted = torch.max(y_predict.data, dim=1)
        return loss, predicted

    def _train_an_epoch(self, train_loader, epoch_id):
        self._model.train()
        total = 0
        correct = 0
        for batch_id, (images, labels) in enumerate(train_loader):
            if self._config['use_cuda']:
                images, labels = images.cuda(), labels.cuda()
            loss, predicted = self._train_single_batch(images, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100.0
        self.train_accuracies.append(accuracy)
        print('Training Epoch: {}, accuracy rate: {:.2f}%'.format(epoch_id, accuracy))

    def train(self, train_dataset):
        self.use_cuda()
        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            data_loader = Construct_DataLoader(train_dataset, self._config['batch_size'])
            self._train_an_epoch(data_loader, epoch_id=epoch)

    def use_cuda(self):
        if self._config['use_cuda']:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()

    def test(self, test_loader):
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if self._config['use_cuda']:
                    images, labels = images.cuda(), labels.cuda()
                outputs = self._model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100.0
        self.test_accuracies.append(accuracy)
        print('Test Accuracy: {:.2f}%'.format(accuracy))

    def plot_accuracies(self):
        plt.plot(self.train_accuracies, label='Training Accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy Over Epochs')
        plt.show()

alexnet_config = {
    'num_epoch': 10,
    'batch_size': 100,
    'lr': 1e-4,
    'l2_regularization': 1e-4,
    'num_classes': 10,
    'device_id': 0,
    'use_cuda': True,
    'model_name': 'AlexNet.ckpt'
}

if __name__ == "__main__":
    alexNet = AlexNet(alexnet_config)
    alexNet.loadModel(map_location='cuda' if alexnet_config['use_cuda'] else 'cpu')
    # for param in alexNet.features.parameters():
    #     param.requires_grad = False  # Freeze the feature layers
    for param in alexNet.features[:-2].parameters():  # Freeze all layers except the last convolutional layer
        param.requires_grad = False
    for param in alexNet.features[-2:].parameters():  # Allow last convolutional layer to be fine-tuned
        param.requires_grad = True
    alexNet.classifier[-1] = nn.Linear(1024, 10)  # Update the final layer for MNIST classification
    if alexnet_config['use_cuda']:
        alexNet.cuda()

    trainer = Trainer(model=alexNet, config=alexnet_config)
    trainer.train(train_set)
    trainer.save()
    trainer.test(Construct_DataLoader(val_set, alexnet_config['batch_size']))
    trainer.plot_accuracies()
