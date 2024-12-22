import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_o = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure the original images are resized to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset function
def LoadCIFAR10(download=False):
    # Load original CIFAR-10 dataset with the same transform as augmented dataset
    original_train_dataset = torchvision.datasets.CIFAR10(root='../CIFAR10', train=True, transform=transform_o, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root='../CIFAR10', train=False, transform=transform_o)

    # Load augmented CIFAR-10 dataset
    augmented_train_dataset = torchvision.datasets.CIFAR10(root='../CIFAR10', train=True, transform=transform, download=download)

    # Combine the original and augmented datasets
    combined_train_dataset = ConcatDataset([original_train_dataset, augmented_train_dataset])

    return combined_train_dataset, test_dataset

# Define the DataLoader construction function
def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

# Define the AlexNet model
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

# Define the Trainer class
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
            if self._config['use_cuda'] is True:
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
        if self._config['use_cuda'] is True:
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

# Define the configuration parameters
alexnet_config = {
    'num_epoch': 10,
    'batch_size': 64,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'num_classes': 10,
    'device_id': 0,
    'use_cuda': True,
    'model_name': 'AlexNet_224.ckpt'
}

if __name__ == "__main__":
    train_dataset, test_dataset = LoadCIFAR10(True)
    alexNet = AlexNet(alexnet_config)
    trainer = Trainer(model=alexNet, config=alexnet_config)
    trainer.train(train_dataset)
    trainer.save()
    trainer.test(Construct_DataLoader(test_dataset, alexnet_config['batch_size']))
    trainer.plot_accuracies()
