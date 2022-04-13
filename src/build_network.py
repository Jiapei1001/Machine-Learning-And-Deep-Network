from random import shuffle
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# initial training parameters
batch_size_train = 64
batch_size_test = 1000

# settings, including random seed
random_seed = 37
torch.manual_seed(random_seed)

torch.backends.cudnn.enabled = False

# load MINIST data
def loadMNIST():
    transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])
    minist_train_set = torchvision.datasets.MNIST('../MINIST/train', train=True, download=True, transform=transform)
    minist_test_set = torchvision.datasets.MNIST('../MINIST/test', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(minist_train_set, batch_size = batch_size_train, shuffle = True)
    test_loader = torch.utils.data.DataLoader(minist_test_set, batch_size = batch_size_test, shuffle = True)

    return train_loader, test_loader

# the entry function to build the network
def main():
    # A. Get the MNIST digit data set
    train_loader, test_loader = loadMNIST()

    # look at a batch of train data
    training_examples = enumerate(train_loader)
    batch_idx, (training_examples_data, training_examples_targets) = next(training_examples)

    print(training_examples_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(training_examples_data[i][0], cmap='viridis', interpolation='none')
        plt.title("Data Label: {}".format(training_examples_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    fig

    


if __name__ == "__main__":
	main()