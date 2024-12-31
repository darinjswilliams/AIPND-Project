import argparse
import json

import torch
import numpy as np
from typing import Dict

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

def load_checkpoint(filepath: str = 'checkpoint.pth'):
    # load configurtion from checkpooint.pth
    check_point = torch.load(filepath)

    # assign model
    model = models.vgg16(pretrained=True)
    model.class_to_idx = check_point['class_to_idx']
    model.classifier = check_point['classifier']
    model.load_state_dict(check_point['state_dict'])

    learning_rate = check_point['learning_rate']
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    optimizer = optimizer.load_state_dict(check_point['optimizer'])
    input_size = check_point['input_size']
    hidden_layer = check_point['hidden_units']
    output_size = check_point['output_size']

    # Hyper parameters
    epochs = check_point['epochs']

    return model


def save_checkpoint(image_datasets: Dict,
                    model: nn.Module,
                    classifier: nn.Module,
                    optimizer: optim.Optimizer,
                    epochs: int,
                    input_features: int,
                    hidden_layer: int,
                    path: str = 'checkpoint.pth') -> None:
    # Validate the dataset input
    if ('train' not in image_datasets
            or not hasattr(image_datasets['train'], 'class_to_idx')):
        raise ValueError("image_datasets['train'] must exist and"
                         " have a 'class_to_idx' attribute.")

    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'input_size': input_features,
                  'hidden_units': hidden_layer,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs
                  }

    # Use torch to save
    torch.save(checkpoint, path)


def get_cat_to_names(self):
    with open('cat_to_name.json') as datafile:
        cat_to_name = json.load(datafile)
    return cat_to_name


def get_input_args():
    parser = argparse.ArgumentParser(description='Explore Command Line Options for Image Classification')

    parser.add_argument('--dir',  type=str, default='flowers/', help='Directory where images are stored')

    parser.add_argument('--arch', dest='arch', type=str, default='vgg', help='Model Architecture')

    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='Learning Rate')

    parser.add_argument('--hidden_unit', dest='hidden_unit', type=int, default=512, help='Hidden Unit')

    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of Epochs')

    parser.add_argument('--save_dir', dest='save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')

    parser.add_argument('--gpu', dest='gpu', type=bool, default=False, help='Use GPU if available')

    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return top K most likely classes')

    parser.add_argument('--category_names',  type=str, default='cat_to_name.json', help='Mapping of categories to real names in dictionary format')

    return parser.parse_args()

if __name__ == '__main__':
    get_input_args()

