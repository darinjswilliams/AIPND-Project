import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse
import json
from typing import Dict
from utilities import get_cat_to_names

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from torchvision import datasets, models, transforms


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


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze_(0)
    model.eval()
    # move to device
    image, model = image.to(device), model.to(device)
    cat_to_name = get_cat_to_names()

    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)

        top_ps, top_classes = ps.topk(topk, dim=1)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU if available')

if __name__ == '__main__':
    main()