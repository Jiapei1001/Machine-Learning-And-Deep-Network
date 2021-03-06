from random import shuffle
from turtle import forward
from black import main
from numpy import block, dtype
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

import cv2
import sys
import os
import csv

import network_model

# parameters
greek_symbols_path = '../greek_symbols'
greek_symbols_csv = '../greek_symbols/greek_symbols.csv'
greek_symbols_category_csv = '../greek_symbols/greek_symbols_category.csv'
handwritten_greek_symbols_path = '../handwritten_greek_symbols'

# greek symbols data
imgs, labels = [], []
valid_images = [".jpg", ".gif", ".png", ".tga"]

embedding_spaces = []


# helper function to load images from a file path
def loadHandwrittenSymbols(path):
    imgs = []
    processed_imgs = []
    labels = []
    # Reference - https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python-to-list-or-dictionary
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        i = cv2.imread(path + '/' + f)
        # gray
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        imgs.append(i)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            i, 80, 255, cv2.THRESH_BINARY)

        # scale down to 28x28
        blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (28, 28))
        # invert image intensities
        # blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)

        l = f.split('.')[0]

        processed_imgs.append(blackAndWhiteImage)
        labels.append(l)

    return imgs, processed_imgs, labels

# helper function to load images from a file path
def loadSymbols(path):
    # Reference - https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python-to-list-or-dictionary
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        i = cv2.imread(path + '/' + f)
        # scale down to 28x28
        # i = cv2.resize(i, (28, 28))
        # gray
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        # invert image intensities
        i = cv2.bitwise_not(i)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            i, 52, 255, cv2.THRESH_BINARY)

        # scale down to 28x28
        blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (28, 28))

        l = f.split('_')[0]

        imgs.append(blackAndWhiteImage)
        labels.append(l)

# write mat to csv
def writeOutToCSV():
    with open(greek_symbols_csv, 'w', newline='') as f, open(greek_symbols_category_csv, 'w', newline='') as c:
        writer = csv.writer(f)
        category = csv.writer(c)

        for i in range(len(imgs)):
            writer.writerow(numpy.asarray(imgs[i]).flatten())
            if (labels[i] == 'alpha'):
                category.writerow('0')
            elif (labels[i] == 'beta'):
                category.writerow('1')
            elif (labels[i] == 'gamma'):
                category.writerow('2')

# helper function to create an embedding space using the images
def createEmbeddingSpace(truncated_network):
    for img in imgs:
        curr = truncated_network(
            torch.tensor(img.reshape(1, 1, 28, 28), dtype=torch.float32)).detach().numpy().flatten()
        embedding_spaces.append(curr)

# helper function to get square sum distances
def squareSumDist(a, b):
    # Reference - https://stackoverflow.com/questions/2284611/sum-of-square-differences-ssd-in-numpy-scipy
    dist = numpy.sum((a - b)**2)
    return round(dist, 2)

# entry function to build a network as an embedding space
def main(argv):
    # load greek symbols
    loadSymbols(greek_symbols_path)
    writeOutToCSV()

    # load network weights to truncated network model
    loaded_model_weights = torch.load(network_model.network_model_location)
    truncated_network = network_model.TruncatedNetworkThree()
    truncated_network.eval()
    truncated_network.load_state_dict(loaded_model_weights)

    # create embedding space
    createEmbeddingSpace(truncated_network)

    # project the greek symbols into the embedding space
    # Pandas Data Frame
    # Reference - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html#pandas.DataFrame.from_dict
    images_dict = {'embedding': embedding_spaces, 'label': labels}
    df = pd.DataFrame.from_dict(images_dict)
    # Shuffle
    # Reference - https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # df = df.sample(frac=1)

    # select an example from each set
    alpha = df[df.label == 'alpha'].sample(1).iloc[0].embedding
    beta = df[df.label == 'beta'].sample(1).iloc[0].embedding
    gamma = df[df.label == 'gamma'].sample(1).iloc[0].embedding

    for i, r in df.iterrows():
        features = r.embedding.flatten()
        alpha_dist = squareSumDist(alpha, features)
        beta_dist = squareSumDist(beta, features)
        gamma_dist = squareSumDist(gamma, features)

        print('Index: {},\t Label: {},\t Alpha Dist: {:.2f},\t\t Beta Dist: {:.2f},\t\t Gamma Dist: {:.2f}'.format(
            i, r['label'], alpha_dist, beta_dist, gamma_dist))

    # load handwritten greek symbols
    original_imgs, hw_imgs, hw_labels = loadHandwrittenSymbols(
        handwritten_greek_symbols_path)

    # load into truncated model
    with torch.no_grad():
        fig = plt.figure()
        # predicted images
        for i in range(len(hw_imgs)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()

            embed = truncated_network(torch.tensor(
                hw_imgs[i].reshape(1, 1, 28, 28), dtype=torch.float32)).detach().numpy().flatten()

            min_dist = sys.maxsize
            prediction = ''
            for _, r in df.iterrows():
                features = r.embedding.flatten()
                a = squareSumDist(embed, features)
                if (a < min_dist):
                    min_dist = a
                    prediction = r.label

            plt.imshow(hw_imgs[i])
            plt.title("Label: {}, Prediction: {}".format(
                hw_labels[i], prediction))
            plt.xticks([])
            plt.yticks([])

        # original images
        for i in range(3, 6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(original_imgs[i - 3], cmap='Greys')
            plt.xticks([])
            plt.yticks([])

        plt.show()
        fig


if __name__ == "__main__":
    main(sys.argv)
