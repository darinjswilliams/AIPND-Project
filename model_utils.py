import json
import os
from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import DATA_ROOT
from makeparse import make_parser
from validation_utilis import get_device, model_validation

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


"""
Returns the required transformations for either training or validation/testing.

Args:
    train (bool): If True, includes training-specific transformations like random rotation.
                  If False, only includes validation/testing transformations.

Returns:
    torchvision.transforms.Compose: The composed transformation pipeline.
"""
def get_transformation(training=True):
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

'''   
    Lets determine the batch size by device we are training model
    if We are training on GPU than lets set batch size to 64
    if training is on CPU than batch size is 32 
'''
def get_dataloaders(train_datasets, valid_datasets, test_datasets, input_args=None):

    device = get_device(input_args.gpu)

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


def get_cat_to_names(input_args='cat_to_name.json'):

    if not os.path.isfile(input_args):
        raise FileNotFoundError(f"File path {input_args} does not exist.")

    with open(input_args) as datafile:
        cat_to_name = json.load(datafile)
    return cat_to_name


