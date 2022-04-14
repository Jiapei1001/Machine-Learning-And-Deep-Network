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

import network_model


# initial training parameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
network_model_location = 'network/model.pt'

# settings: random seed
random_seed = 37
torch.manual_seed(random_seed)
# settings: turn off CUDA to make the process repeatable
torch.backends.cudnn.enabled = False


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


# Step A. Get the MNIST digit data set
train_loader, test_loader, mnist_train_set, mnist_test_set = loadMNIST()

# look at a batch of train data
training_examples = enumerate(train_loader)
batch_idx, (training_examples_data, training_examples_targets) = next(
    training_examples)

print(training_examples_data.shape)

# plot 6 images
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(training_examples_data[i][0],
               cmap='viridis', interpolation='none')
    plt.title("Data Label: {}".format(training_examples_targets[i]))
    plt.xticks([])
    plt.yticks([])

# don't block following computation process
plt.show(block=False)
fig


# Step B. Settings to make the network repeatable

# Step C. Build the network model
network = network_model.Network()
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate, momentum=momentum)

# Step D. Train the model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    '''
    Train - entry function to train the network model
    '''
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))


def test():
    '''
    evaluation of the training model
    '''
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# entry to Train
test()
for epoch in range(1, n_epochs + 1):
    train(epoch=epoch)
    test()


# plot the training and testing accuracy
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show(block=False)
fig


# Step E. Save the network to a file
torch.save(network.state_dict(), network_model_location)


# Step F. Read the network and run it on the test set
loaded_model_weights = torch.load(network_model_location)
# Evaluation mode
# In training mode the dropout layer randomly sets node values to zero.
# In evaluation mode, it multiplies each value by 1-dropout rate so the same pattern will generate the same output each time.
network.eval()
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

plt.show()
fig
