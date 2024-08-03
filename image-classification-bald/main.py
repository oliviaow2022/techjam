"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from copy import deepcopy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader, Subset

from acquisition import *
from util import *


lr = 0.001
acquisition_batch_size = 64
train_batch_size = 64
num_train = 5000
num_pool = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10) # 10 classes for mnist

        # self.layers = nn.Sequential(
        #     nn.Linear(784, 2048),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(2048, 128),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(128, 10))

    def forward(self, x, return_logits=False):
        # x = torch.flatten(x, 1)
        # logits = self.layers(x)
        logits = self.model(x)
        if return_logits:
            return logits
        else:
            return F.softmax(logits, 1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, return_logits=True)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, return_logits=True)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = float(correct) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))

    return accuracy

def active(model, acquirer, device, data, optimizer):
    train_data, pool_data, test_data = data

    test_accuracies = []
    while len(pool_data) > 0:
        print(f'Acquiring {acquirer.__class__.__name__} batch. Pool size: {len(pool_data)}')
        # get the indices of the best batch of data
        batch_indices = acquirer.select_batch(model, pool_data)
        # move that data from the pool to the training set
        move_data(batch_indices, pool_data, train_data)
        # train on it
        train_loader = DataLoader(train_data,
            batch_size=train_batch_size, pin_memory=True, shuffle=True)
        train(model, device, train_loader, optimizer, 0)

        # test the accuracy
        test_loader = DataLoader(test_data,
            batch_size=train_batch_size, pin_memory=True, shuffle=True)
        accuracy = test(model, device, test_loader)
        test_accuracies.append(accuracy)

    return test_accuracies


if __name__ == '__main__':
    # set up the GPU if one exists
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    
    subset_indices = np.random.choice(len(dataset), size=num_train+num_pool, replace=False)
    train_indices = subset_indices[:num_train]
    pool_indices = subset_indices[-num_pool:]
    train_data = Subset(dataset, train_indices)
    test_data = datasets.MNIST('data', train=False,
                       transform=transform)
    # pool_data = Subset(dataset, pool_indices)
    # test_size = int(0.2 * len(dataset))
    # train_size = len(dataset) - test_size
    # train_data, test_data = random_split(dataset, [train_size, test_size])
    
    pretrain_loader = DataLoader(train_data,
        batch_size=train_batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data,
        batch_size=train_batch_size, pin_memory=True, shuffle=True)

    # init the model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train the model and test after each epoch
    # Train and test the model for a few epochs before active learning
    for epoch in range(1, 2):
        train(model, device, pretrain_loader, optimizer, epoch)
        # test(model, device, test_loader)\
        starting_accuracy = test(model, device, test_loader)\

    pre_acquisition_model_state = model.state_dict()

    for acquisition_strategy in [Random, BALD, BatchBALD, Entropy, LeastConfidence, DiversitySampling]:
        accuracy_list = [starting_accuracy]
        # reset the model
        model.load_state_dict(deepcopy(pre_acquisition_model_state))
        # init the acquirer
        acquirer = acquisition_strategy(acquisition_batch_size, device)
        # and an optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # get all the data
        train_data = Subset(dataset, train_indices)
        pool_data = Subset(dataset, pool_indices)
        data = (train_data, pool_data, test_data)
        # train the model with active learning
        accuracies = active(model, acquirer, device, data, optimizer)
        print("accuracies:",accuracies)
        # plot the accuracies
        plt.plot(accuracies, label=acquisition_strategy.__name__)

    plt.xlabel('Acquisition Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()