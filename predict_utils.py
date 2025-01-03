import os
import pathlib
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch import optim, nn

from config import DATA_ROOT
from model_utils import get_model_architecture
from validation_utilis import get_device

'''
Get a random image from the flowers directory
using the Glob Search Pattern to look for jpg images

Returns:
    image_path: Path to the image

'''
def random_select_image():

    image_path = pathlib.Path(DATA_ROOT)

    # Get all images under flowers
    all_images = list(image_path.glob('**/*.[jJ][pP][gG]'))
    print(len(all_images))
    # Validate that images exists
    if not all_images:
        raise ValueError("Did not Find any Images")

    rand_image_path = np.random.choice(all_images)
    image = Image.open(rand_image_path)

    return str(rand_image_path), image


def load_checkpoint(args):
    # load configurtion from checkpooint.pth

    device = get_device(args.gpu)

    '''
      Lets make sure we get the correct device we process model on
      to avoid errors and days of debugging
      map_location is a torch.device object or a string containing a device tag, it indicates 
      the location where all tensors should be loaded.
    '''
    file_path = os.path.join(args.save_dir, args.checkpoint)
    if pathlib.Path(str(file_path)).exists() is False:
        print("The CheckPoint path does not exist:")
        exit(1)

    check_point = torch.load(f=str(file_path), map_location=device)


    # assign model
    model = get_model_architecture(args)
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


'''
    The following information that was previously saved  is coming from checkpoint

      'input_size': input_features,
      'hidden_units': hidden_layer,
      'output_size': 102,
      'class_to_idx': model.class_to_idx,
      'state_dict': model.state_dict(),
      'classifier': classifier,
      'learning_rate': learning_rate,
      'optimizer': optimizer.state_dict(),
      'epochs': epochs 
      
      
    Inserted a validation check where dataset information is coming from
    if the dataset does not have train included in the dictionary than
    it will raise a Value Error
      
 '''
def save_checkpoint(image_datasets: Dict,
                    model: nn.Module,
                    classifier: nn.Module,
                    optimizer: optim.Optimizer,
                    epochs: int,
                    input_features: int,
                    hidden_layer: int,
                    learning_rate: float,
                    file_path) -> None:
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
                  'learning_rate': learning_rate,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs
                  }

    # Let check to see if the path exist
    if file_path.save_dir is not None and file_path.checkpoint is not None :
       create_directory(file_path.save_dir)
       file_path = os.path.join(file_path.save_dir, file_path.checkpoint)
       torch.save(checkpoint, str(file_path))
    else:
        torch.save(checkpoint, str(file_path.checkpoint))

def create_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return True

if __name__ == '__main__':
    image, image_path = random_select_image()
    print(image_path)
    print(image)


