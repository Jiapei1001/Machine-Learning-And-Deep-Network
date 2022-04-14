from random import shuffle
from turtle import forward
from numpy import block

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

import cv2

import network_model

# parameters
network_model_location = 'network/model.pt'

# load in network model
loaded_model_weights = torch.load(network_model_location)
network = network_model.Network()
network.eval()
network.load_state_dict(loaded_model_weights)

# Analyze A - Analyze the first convolution layer
fig = plt.figure()
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.tight_layout()
    plt.imshow(network.conv1.weight[i].detach().reshape(5, 5),
               cmap='viridis', interpolation='none')
    plt.title("Filter channel: {}".format(i + 1))
    plt.xticks([])
    plt.yticks([])

plt.show(block=False)
fig

# Analyze B - Show effects after applying first convolution layer to image
# Load image from training set
transform = torchvision.transforms.Compose(

    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
# MNIST train set
mnist_train_set = torchvision.datasets.MNIST(
    '../MINIST/train', train=True, download=True, transform=transform)

image = mnist_train_set.data[0].reshape(28, 28)

# don't need to calculate gradients
with torch.no_grad():
    fig = plt.figure()
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(cv2.filter2D(image.numpy(), -1, network.conv1.weight[i].detach().reshape(5, 5).numpy()),
                   cmap='viridis', interpolation='none')
        plt.title("Filter channel: {}".format(i + 1))
        plt.xticks([])
        plt.yticks([])

    plt.show(block=False)
    fig


# Analyze C - Show effects after a truncated network
# Weights of the model are still the same,
# but only the first two layers will be used if you apply the model to some data.
loaded_model_weights = torch.load(network_model_location)
truncated_network_1 = network_model.TruncatedNetworkOne()
truncated_network_1.eval()
truncated_network_1.load_state_dict(loaded_model_weights)

# output have 10 channels that are 12x12 in size, after conv1
# don't need to calculate gradients
with torch.no_grad():
    fig = plt.figure("Truncated Network w. Conv #1")
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(truncated_network_1(image.reshape(1, 1, 28, 28).float()).detach().numpy().reshape(10, 12, 12)[i],
                   cmap='viridis', interpolation='none')
        plt.title("Filter channel: {}".format(i + 1))
        plt.xticks([])
        plt.yticks([])

    plt.show(block=False)
    fig


truncated_network_2 = network_model.TruncatedNetworkTwo()
truncated_network_2.eval()
truncated_network_2.load_state_dict(loaded_model_weights)

# output have 20 channels that are 4x4 in size, after conv1 & conv2
# don't need to calculate gradients
with torch.no_grad():
    fig = plt.figure("Truncated Network w. Conv #1 & #2")
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        plt.imshow(truncated_network_2(image.reshape(1, 1, 28, 28).float()).detach().numpy().reshape(20, 4, 4)[i],
                   cmap='viridis', interpolation='none')
        plt.title("Filter channel: {}".format(i + 1))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    fig
