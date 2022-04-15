from cProfile import label
from random import shuffle
from turtle import forward
from numpy import block

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

import network_model

# initial parameters
log_interval = 10
momentum = 0.5

# settings: random seed
random_seed = 37
torch.manual_seed(random_seed)
# settings: turn off CUDA to make the process repeatable
torch.backends.cudnn.enabled = False

# load MINIST data, based on variations on batch_sizes
def loadMNIST(batch_size) -> torch.utils.data.DataLoader:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

    # MNIST train & test set
    mnist_train_set = torchvision.datasets.MNIST(
        '../MINIST/train', train=True, download=True, transform=transform)
    mnist_test_set = torchvision.datasets.MNIST(
        '../MINIST/test', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        mnist_train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, mnist_train_set, mnist_test_set

# entry function to train and test
def train_and_test(n_epochs, learning_rate, batch_size, momentum):
    # network model
    network = network_model.Network()
    optimizer = optim.SGD(network.parameters(),
                          lr=learning_rate, momentum=momentum)

    train_loader, test_loader, mnist_train_set, mnist_test_set = loadMNIST(
        batch_size)

    # traning results
    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(0, network, test_loader, test_loss)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader,
              optimizer, train_loss, train_counter)

        test(epoch, network, test_loader, test_loss)

    return train_loss, train_counter, test_loss, test_counter

# entry function to train the model
def train(epoch, network, train_loader, optimizer, train_loss, train_counter):
    # Train
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % log_interval == 0):
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))

            train_loss.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))


# entry function to test the result
def test(epoch, network, test_loader, test_loss):
    network.eval()
    temp_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            temp_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        temp_loss /= len(test_loader.dataset)
        test_loss.append(temp_loss)
        print('Train Epoch: {}, Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch,
            temp_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# entry function to compare different batch sizes
def dimensionBatchSize():
    n_epochs = 3
    batch_size = [64, 128, 256]
    learning_rate = 0.01

    train_losses = []
    train_counters = []
    test_losses = []
    test_counters = []
    colors = ['b', 'g', 'y']

    for i in range(3):
        print('\nConfigurations - Total Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(
            n_epochs, batch_size[i], learning_rate))

        train_loss, train_counter, test_loss, test_counter = train_and_test(
            n_epochs, learning_rate, batch_size[i], momentum=momentum)
        train_losses.append(train_loss)
        train_counters.append(train_counter)
        test_losses.append(test_loss)
        test_counters.append(test_counter)

    fig = plt.figure()
    idx = 0
    for train_loss, train_counter, test_loss, test_counter, color in zip(train_losses, train_counters, test_losses, test_counters, colors):
        plt.plot(train_counter, train_loss, color=color,
                 label='bs: ' + str(batch_size[idx]))
        plt.scatter(test_counter, test_loss, color=color)
        idx += 1
    plt.legend()
    plt.title('Batch Size variations')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show(block=False)
    fig

# entry function to compare different learning rates
def dimensionLearningRate():
    n_epochs = 3
    batch_size = 64
    learning_rate = [0.1, 0.01, 0.001]

    train_losses = []
    train_counters = []
    test_losses = []
    test_counters = []
    colors = ['b', 'g', 'y']

    for i in range(3):
        print('\nConfigurations - Total Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(
            n_epochs, batch_size, learning_rate[i]))

        train_loss, train_counter, test_loss, test_counter = train_and_test(
            n_epochs, learning_rate[i], batch_size, momentum=momentum)
        train_losses.append(train_loss)
        train_counters.append(train_counter)
        test_losses.append(test_loss)
        test_counters.append(test_counter)

    fig = plt.figure()
    idx = 0
    for train_loss, train_counter, test_loss, test_counter, color in zip(train_losses, train_counters, test_losses, test_counters, colors):
        plt.plot(train_counter, train_loss, color=color,
                 label='lr: ' + str(learning_rate[idx]))
        plt.scatter(test_counter, test_loss, color=color)
        idx += 1
    plt.legend()
    plt.title('Learning rate viriations')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show(block=False)
    fig

# entry function to compare different epochs
def dimensionEpochs():
    n_epochs = [2, 4, 6]
    batch_size = 64
    learning_rate = 0.01

    train_losses = []
    train_counters = []
    test_losses = []
    test_counters = []
    colors = ['b', 'g', 'y']

    for i in range(3):
        print('\nConfigurations - Total Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(
            n_epochs[i], batch_size, learning_rate))

        train_loss, train_counter, test_loss, test_counter = train_and_test(
            n_epochs[i], learning_rate, batch_size, momentum=momentum)
        train_losses.append(train_loss)
        train_counters.append(train_counter)
        test_losses.append(test_loss)
        test_counters.append(test_counter)

    fig = plt.figure()
    idx = 0
    for train_loss, train_counter, test_loss, test_counter, color in zip(train_losses, train_counters, test_losses, test_counters, colors):
        plt.plot(train_counter, train_loss, color=color,
                 label='ep: ' + str(n_epochs[idx]))
        plt.scatter(test_counter, test_loss, color=color)
        idx += 1
    plt.legend()
    plt.title('Epoch number variations')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    fig


# entry function to start the program
def dimensions(argv):
    print("\n\nDimenision Variation - Batch Size")
    dimensionBatchSize()

    print("\n\nDimenision Variation - Learning Rate")
    dimensionLearningRate()

    print("\n\nDimenision Variation - Number of Epochs")
    dimensionEpochs()


if __name__ == "__main__":
    dimensions(sys.argv)
