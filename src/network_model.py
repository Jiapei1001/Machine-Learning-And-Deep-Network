from random import shuffle
from turtle import forward
from numpy import block
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Network model class
class Network(nn.Module):
    # constructing Network using torch.nn
    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # Visualization: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    def __init__(self) -> None:
        super(Network, self).__init__()
        # A convolution layer with in_channels, out_channels, and kernel_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 50%
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x) -> torch.Tensor:
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = self.conv2_drop(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Truncated Network model class, with only the first convolution layers
class TruncatedNetworkOne(nn.Module):
    def __init__(self) -> None:
        super(TruncatedNetworkOne, self).__init__()
        # A convolution layer with in_channels, out_channels, and kernel_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 50%
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x) -> torch.Tensor:
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        return x


# Truncated Network model class, with only the first & second convolution layers
class TruncatedNetworkTwo(nn.Module):
    def __init__(self) -> None:
        super(TruncatedNetworkTwo, self).__init__()
        # A convolution layer with in_channels, out_channels, and kernel_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 50%
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x) -> torch.Tensor:
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = self.conv2_drop(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        return x
