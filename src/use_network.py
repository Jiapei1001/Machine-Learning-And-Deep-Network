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

import sys
import os
import cv2

import network_model


# initial training parameters
batch_size_train = 64
batch_size_test = 1000
valid_images = [".jpg", ".gif", ".png", ".tga"]
handwritten_numbers_path = '../handwritten_numbers'


# load MINIST data
def loadMNIST() -> torch.utils.data.DataLoader:
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
        mnist_train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test_set, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader, mnist_train_set, mnist_test_set


# helper function to load images from a file path
def loadSymbols(path):
    imgs = []
    labels = []
    # Reference - https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python-to-list-or-dictionary
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        i = cv2.imread(path + '/' + f)
        # gray
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(
            i, 160, 255, cv2.THRESH_BINARY)

        # scale down to 28x28
        blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (28, 28))
        # invert image intensities
        # blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)

        l = f.split('.')[0]

        imgs.append(blackAndWhiteImage)
        labels.append(l)

    return imgs, labels


# entry function to use the network
def main(argv):
    train_loader, test_loader, mnist_train_set, mnist_test_set = loadMNIST()

    # Step F. Read the network and run it on the test set
    network = network_model.Network()
    loaded_model_weights = torch.load(network_model.network_model_location)
    # Evaluation mode
    # In training mode the dropout layer randomly sets node values to zero.
    # In evaluation mode, it multiplies each value by 1-dropout rate so the same pattern will generate the same output each time.
    network.eval()
    network.load_state_dict(loaded_model_weights)

    # reshape to 10 images, each 1 channel, size as 28 * 28
    eval_in = mnist_test_set.data[2:12].reshape(10, 1, 28, 28).float()
    eval_res = network(eval_in)
    predictions = torch.argmax(eval_res, dim=1)

    # plot 10 images
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(mnist_test_set.data[i + 2],
                   cmap='viridis', interpolation='none')
        plt.title("Label: {}, Prediction: {}".format(
            mnist_test_set.targets[i + 2], predictions[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show(block=False)
    fig

    # Step G. Test the network on new inputs
    imgs, labels = loadSymbols(handwritten_numbers_path)

    # plot 10 images
    with torch.no_grad():
        fig = plt.figure()
        for i in range(len(imgs)):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()

            eval_res = network(torch.tensor(
                imgs[i].reshape(1, 1, 28, 28), dtype=torch.float32))
            prediction = torch.argmax(eval_res, dim=1).item()

            plt.imshow(imgs[i])
            plt.title("Label: {}, Prediction: {}".format(
                labels[i], prediction))
            plt.xticks([])
            plt.yticks([])

        plt.show()
        fig


if __name__ == "__main__":
    main(sys.argv)
