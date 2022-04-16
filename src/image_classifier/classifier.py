import config as cg
from torchvision import models
import numpy as np
import argparse
import torch
import cv2

import glob
from PIL import Image

import sys
import math

# parameters
FONT_SCALE = 1.6e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 1.6e-3  # Adjust for larger thickness in all images

# initial models
MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "inception": models.inception_v3(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
    "resnet": models.resnet50(pretrained=True)
}

# preprocess images to feed into opencv
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cg.IMAGE_SIZE, cg.IMAGE_SIZE))
    # normalize the pixels to the range of [0, 1]
    img = img.astype("float32") / 255.0

    # subtract mean, divide by standard deviation
    img -= cg.MEAN
    img /= cg.STD
    # set "channels first" ordering, add batch dimension
    # moving the channels dimension to the front of the array
    # it is the default channel ordering method that PyTorch expects
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return img

# calculate label and probability for an image
def getLabelAndProbability(model, img):
    processed_img = preprocess(img)

    # convert image to torch tensor
    processed_img = torch.from_numpy(processed_img)
    processed_img = processed_img.to(cg.DEVICE)

    # load labels
    print("loading the library of labels...")
    labels = dict(enumerate(open(cg.IN_LABELS)))

    # classify image
    print("classifying image with model {}...".format(model.__class__.__name__))
    options = model(processed_img)
    candidates = torch.nn.Softmax(dim=-1)(options)
    sortedProbabilities = torch.argsort(candidates, dim=-1, descending=True)

    # top 5 ranking labels
    for (i, idx) in enumerate(sortedProbabilities[0, :5]):
        print("{}, {}: {:.2f}%".format(
            i, labels[idx.item()].strip(), sortedProbabilities[0, idx.item()] * 100))

    print("\n")

    # display
    (label, prob) = (
        labels[candidates.argmax().item()], candidates.max().item())

    return (label, prob)

# entry point to start the program
def main(argv):
    cmd = argparse.ArgumentParser()
    cmd.add_argument("-i", "--image", required=True,
                     help="path to images")
    cmd.add_argument("-m", "--model", type=str, default="vgg16",
                     choices=["vgg16", "densenet", "vgg19",
                              "resnet", "inception", ],
                     help="network to use")
    args = vars(cmd.parse_args())

    # load model
    print("loading model {}...".format(args["model"]))
    model = MODELS[args["model"]].to(cg.DEVICE)

    # evaluation mode
    model.eval()

    # load image
    print("loading image {}...".format(args["image"]))
    imdir = args["image"]
    files = glob.glob(imdir + "*.jpg")

    images = []
    oris = []
    for i in files:
        img = cv2.imread(i)
        images.append(img)
        oris.append(img.copy())

    labels, probabilities = [], []

    for img in images:
        label, prob = getLabelAndProbability(model, img)
        labels.append(label)
        probabilities.append(prob)

    for i in range(len(images)):
        height, width, _ = images[i].shape
        font_scale = min(width, height) * FONT_SCALE
        thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
        pos = int(height * 0.1)
        cv2.putText(oris[i], "Label: {}, {:.2f}%".format(labels[i].strip(), probabilities[i] * 100),
                    (50, pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.imshow("Classification #{}".format(i), oris[i])

    cv2.waitKey(0)


if __name__ == "__main__":
    main(sys.argv)
