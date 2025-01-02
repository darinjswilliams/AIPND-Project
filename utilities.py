import argparse
import json
import os

import torch
import numpy as np
import pathlib

from typing import Dict

from pygments.lexer import default
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
from pathlib import Path

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import datasets, models, transforms


# Paths to the root of the project and the `data` subfolder.
PROJECT_ROOT = pathlib.Path(__name__).parent.resolve()
DATA_ROOT = PROJECT_ROOT / 'flowers'

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

    # Validate that images exists
    if not all_images:
        raise ValueError("Did not Find any Images")

    rand_image_path = np.random.choice(all_images)

    return str(rand_image_path)

'''
Input: 
    args: Command Line Arguments
Returns:
    model: The model architecture based on the input argument
'''
def get_model_architecture(args):

    model_name = model_validation(args)

    return getattr(models, model_name)(weights=True)

'''
Input: 
    file_path: Path to the directory where images are stored
Returns:
    train_datasets, valid_datasets, test_datasets, image_datasets
'''
def get_datasets(file_path):

    train_dir, valid_dir, test_dir = get_process_path(file_path)

    train_transform = get_transformation(True)
    test_transform = get_transformation(False)

    test_datasets = datasets.ImageFolder(root=test_dir, transform=test_transform)
    train_datasets = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_datasets = datasets.ImageFolder(root=valid_dir, transform=test_transform)

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=train_transform),
        'validation': datasets.ImageFolder(root=valid_dir, transform=test_transform),
        'test': datasets.ImageFolder(root=test_dir, transform=test_transform)
    }

    return train_datasets, valid_datasets, test_datasets, image_datasets


''''
Input: 
    file_path: Path to the directory where images are stored
Returns:
    train_dir, valid_dir, test_dir
'''
def get_process_path(file_path: str):

    root_dir = file_path.split('/')[-1]
    if root_dir in ['train', 'valid', 'test']:
        train_dir = os.path.join(DATA_ROOT, 'train')
        valid_dir = os.path.join(DATA_ROOT, 'valid')
        test_dir = os.path.join(DATA_ROOT, 'test')
        return train_dir, valid_dir, test_dir
    elif Path(file_path).exists():
        return file_path
    else:
        raise FileNotFoundError(f"File path {file_path} does not exist.")


def get_transformation(training=True):
    """
    Returns the required transformations for either training or validation/testing.

    Args:
        train (bool): If True, includes training-specific transformations like random rotation.
                      If False, only includes validation/testing transformations.

    Returns:
        torchvision.transforms.Compose: The composed transformation pipeline.
    """
    # Common transformations (applied to both training and validation/testing datasets)
    base_transforms = [
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize
    ]

    if training:
        # Add data augmentation transformations for training
        train_transforms = [transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224)]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        # Return only the common transformations for validation/testing
        test_transformations = [transforms.Resize(256),
                                transforms.CenterCrop(224)] # Center crop to 224x224]
        return transforms.Compose(test_transformations + base_transforms)


def get_dataloaders(train_datasets, valid_datasets, test_datasets):

    device = get_device()

    # Lets determine the batch size by device we are training model
    # if We are training on GPU than lets set batch size to 64
    # if training is on CPU than batch size is 32
    if device == 'cpu':
        batch_size = 32
    else:
        batch_size = 64

    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

def setup_hyper_params(model, model_name: str, hidden_units: int, learning_rate: float):

    for param in model.parameters():
        param.requires_grad = False

    input_size = 0

    if model_name == 'vgg16':
        input_size = model.classifier[0].in_features
    elif model_name == 'alexnet':
        input_size = model.classifier[1].in_features
    elif model_name == 'densenet121':
        input_size = model.classifier.in_features
    elif model_name == 'googlenet':
        input_size = model.fc.in_features

    # define our new classifier
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))

    # Attach classifier to model
    model.classifier = classifier

    # Loss
    criterion = nn.NLLLoss()

    # Optimizer, use  parameters from model.fc and  learning rate
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

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
 '''
def load_checkpoint(args, filepath: str = 'checkpoint.pth'):
    # load configurtion from checkpooint.pth

    device = get_device()

    '''
      Lets make sure we get the correct device we process model on
      to avoid errors and days of debugging
      map_location is a torch.device object or a string containing a device tag, it indicates 
      the location where all tensors should be loaded.
    '''
    check_point = torch.load(f=filepath, map_location=device)

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
                  'learning_rate': learning_rate,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs
                  }

    # Use torch to save
    torch.save(checkpoint, path)


def get_cat_to_names(input_args='cat_to_name.json'):
    with open(input_args) as datafile:
        cat_to_name = json.load(datafile)
    return cat_to_name

'''
    Use Cuda if available, if not available use cpu
    
    Note:   Cuda is recommended for faster computation,
            using the cpu is very slow and time consuming.
            
           For inference, it request to use gpu specifically if available.
           so the second if statement flow was create
'''
def get_device(use_cuda=None):
    if use_cuda is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_cuda == 'gpu':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


"""
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default location of images
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Save Directory default value 'checkpoint.pth'
      4. Learning Rate Hyperparameter
      5. Hidden Unit Hyperparameter
      6. GPU for training uses cpu as default
      7. Epochs Hyperparameter
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

def make_parser():
    parser = argparse.ArgumentParser(description='Explore Command Line Options for Image Classification')

    parser.add_argument('--dir',  type=str, default=DATA_ROOT, help='Directory where images are stored')

    parser.add_argument('--input', type=pathlib.Path, default=DATA_ROOT, help='Path to File')

    parser.add_argument('--arch', dest='arch', type=str, default='vgg16', help='Model Architecture')

    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='Learning Rate')

    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=512, help='Hidden Unit')

    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of Epochs')

    parser.add_argument('--save_dir', dest='save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')

    parser.add_argument('--gpu', dest='gpu', type=bool, default=False, help='Use GPU if available')

    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return top K most likely classes')

    parser.add_argument('--category_names',  type=str, default='cat_to_name.json', help='Mapping of categories to real names in dictionary format')

    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default='checkpoint.pth', help='Path to checkpoint file')

    return parser.parse_args()

def commandline_validations(in_args):
    # Validate directory
    file_path = os.path.join(DATA_ROOT, in_args.dir)

    if pathlib.Path(file_path).exists():
        print("Directory Path is Valid")
        return file_path
    else:
        print("Directory does not exist or File Does not Exist")
        exit(1)

def model_validation(in_args):
    if not in_args.arch in ['vgg16', 'alexnet', 'densenet121', 'googlenet']:
        print("Model Architecture not supported")
        print("Please Provide {}, {}, {} architecture"
              .format('vgg', 'alexnet', 'densenet121', 'googlenet'))
        exit(1)

    if in_args.hidden_units < 1 or in_args.hidden_units > 1024:
        print("Hidden Units should be between 1 and 1024")
        exit(1)

    print('Processing Model Architecture: {} '.format(in_args.arch.upper( )))
    print('Processing Hidden Layers: {} '.format(in_args.hidden_units))
    return in_args.arch.lower( )


if __name__ == '__main__':
    input_args = make_parser()
    print(input_args)
    print(input_args.arch)
    image_path = random_select_image()
    print(image_path)
    names = get_cat_to_names()
    for k, v in names.items():
        print('Flower Name: {}, Key:{}'.format(v, k))
    # file_path = commandline_validations(input_args)
    # # print(file_path)
    # # print(get_transformation(training=True))
    # # print(get_transformation(training=False))
    # train_datasets, valid_datasets, test_datasets, image_datasets = get_datasets(file_path)
    # # print(get_datasets(file_path))
    # # train_loader, valid_loader, test_loader = get_process_path(file_path)
    # # print('traindir: {}'.format(train_loader))
    # # print('testdir: {}'.format(test_loader))
    # # print('validdir: {}'.format(valid_loader))
    # # train_loader, valid_loader, test_loader = get_dataloaders(train_datasets,
    # #                                                           valid_datasets,
    # #                                                           test_datasets)
    # # print('trainloader: {}'.format(train_loader))
    # # print('validloader: {}'.format(valid_loader,))
    # # print('testloader: {}'.format(test_loader))
    # # print('image dataset: {}'.format(image_datasets))
    # model = get_model_architecture(input_args)
    # print(model)
    # model, criterion, optimizer = setup_hyper_params(model,
    #                                                  input_args.arch,
    #                                                  input_args.hidden_units,
    #                                                  input_args.learning_rate)
    #
    # save_checkpoint(image_datasets=image_datasets,
    #                 model=model,
    #                 classifier=model.classifier,
    #                 optimizer=optimizer,
    #                 epochs=input_args.epochs,
    #                 input_features=model.classifier[0].in_features,
    #                 hidden_layer=input_args.hidden_units,
    #                 learning_rate=input_args.learning_rate,
    #                 path=input_args.checkpoint)
    #
    # model = load_checkpoint(args=input_args,
    #                filepath=input_args.checkpoint)
    # print(model)
    # print(setup_hyper_params(model,
    #                          input_args.arch,
    #                          input_args.hidden_units,
    #                          input_args.learning_rate))
    # print(get_device())
    # # print(get_process_path(input_args))
    # model = get_model_architecture(input_args)
    #
    # print(setup_hyper_params(model, 'vgg16', 512, 0.001))
    # print(get_process_path(input_args))
    # if input_args.category_names:
    #     cat_to_name = get_cat_to_names(input_args.category_names)
    #     print(cat_to_name)
    #
    # if input_args.dir:
    #     print(input_args.dir)
    #
    # model_arch = get_model_architecture(input_args)
    # print(model_arch)

