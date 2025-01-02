import pathlib
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse
import json
from typing import Dict

from matplotlib import pyplot as plt

from utilities import get_cat_to_names, get_device, random_select_image

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from torchvision import datasets, models, transforms
from utilities import make_parser, save_checkpoint, load_checkpoint, get_cat_to_names


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Settings for processing
    shortest_side = 256  # Resize shortest side to 256 pixels
    crop_size = 224  # Center crop size
    normalize_mean = np.array([0.485, 0.456, 0.406])  # Mean for normalization
    normalize_std = np.array([0.229, 0.224, 0.225])  # Std for normalization

    # Open image
    with Image.open(image_path) as img:
        # Resize to maintain aspect ratio with shortest side = 256
        img.thumbnail((shortest_side, shortest_side), Image.ANTIALIAS)

        # Center crop
        left = (img.width - crop_size) / 2
        top = (img.height - crop_size) / 2
        right = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))

        # Convert to NumPy array
        np_image = np.array(img) / 255  # Scale pixel values to [0, 1]

        # Normalize image
        np_image = (np_image - normalize_mean) / normalize_std

        # Transpose dimensions to match PyTorch format (C x H x W)
        np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_path=None, top_k=5, input_args=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = get_device(input_args.gpu)

    image_path = random_select_image() if image_path is None else image_path

    if pathlib.Path(image_path).exists() is False:
        raise ValueError("Path does not exist for image file:")

    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze_(0)

    # Get Model f
    model = load_checkpoint(input_args, input_args.checkpoint)
    model.eval()
    # move to device
    image, model = image.to(device), model.to(device)
    cat_to_name = get_cat_to_names()

    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)

        top_ps, top_classes = ps.topk(top_k, dim=1)

        top_ps = top_ps.tolist()[0]
        top_classes = top_classes.tolist()[0]

        # Invert class_to_idx --> idx_to_class and save to dictionary
        idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}

        # Assign labels to the top_class
        class_labels = [idx_to_class[cls] for cls in top_classes]

        flowers = [cat_to_name[label] for label in class_labels]

    return top_ps, top_classes, flowers


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def print_information(ps=None, className=None, flower_name=None, names=None):

    if ps is not None:
        print("The Name of Flower is : ", flower_name)
        print("The Probability of the Flower is : ", ps)
        print("The Class of the Flower is : ", className)

    if names is not None:
        for k, v in names.items():
            print('Flower Name: {}, Key:{}'.format(v, k))


'''
 Look at the arguments and process 
 1. reads in an image and a checkpoint then prints the most likely
    image class and it's associated probability
    arguments
        --checkpoint - information where model is saved
        --input - the path of image file file

 2.  print out the top K classes along with associated probabilities
     arguments
         -- top_k - the number of top classes to display

 3. load a JSON file that maps the class values to other category names
    arguments
        --category_name

 4. use the GPU to calculate the predictions
    arguments
        --gpu

'''
def main():

    input_args = make_parser()

    if input_args.checkpoint is not None:
        ps, class_name, flower = predict(input_args=input_args,
                                         top_k=input_args.top_k,
                                         image_path=input_args.input)
        print_information(ps=ps, className=class_name, flower_name=flower)

    if input_args.category_names:
        cat_to_name = get_cat_to_names(input_args.category_names)
        print_information(names=cat_to_name)



if __name__ == '__main__':
    main()