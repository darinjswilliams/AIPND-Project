import argparse
import json
from typing import Dict

import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from torchvision import datasets, models, transforms


from torch import nn, optim

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



def get_model_architecture(model_name):
    # update to  getattr(models, arch)(weights=True)
    # vgg16 = models.vgg16(weights=True)
    # alexnet = models.alexnet(weights=True)
    # model = {'alexnet': alexnet, 'vgg16': vgg16}

    return getattr(models, model_name)(weights=True)


def determine_dataset(file_path):

    root_dir = file_path.split('/')[0]

    train_dir = root_dir + '/train'
    valid_dir = root_dir + '/valid'
    test_dir = root_dir + '/test'

    return train_dir, valid_dir, test_dir

def train_model(epochs, trainloaders, validloaders,
                device,
                model,
                criterion,
                optimizer,
                steps,
                running_loss,
                print_everything=5):

    for epoch in range(epochs):

        for images, labels in trainloaders:

            # Accumulate Steps
            steps += 1

            # Move input and label tensors to the GPU or Device
            images, labels = images.to(device), labels.to(device)

            # Zero out gradient
            optimizer.zero_grad()

            # Get log probabilities
            logps = model.forward(images)

            # Get Los from criterion
            loss = criterion(logps, labels)

            # Do backward pass
            loss.backward()

            # Take a step
            optimizer.step()

            # increment running loss
            running_loss += loss.item()

            # Drop out of the training loop and test our network accuracy on test data set

            if steps % print_everything == 0:

                # set our model to evaluation inference mode to turn off dropout
                model.eval()

                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in validloaders:
                        # Move input and label tensors to the GPU or Device
                        images, labels = images.to(device), labels.to(device)

                        # Get log probabilities
                        logps = model(images)

                        # Get Loss from criterion
                        loss = criterion(logps, labels)

                        # Accumulate valid loss
                        valid_loss += loss.item()

                        # calculate accuracy, remember out model is returning logsoftmax
                        # so its the log probabilities of our classes to get actual probabilites
                        ps = torch.exp(logps)

                        # Get the top probabilites and class
                        # Providing 1, will give you the first largest value in your probability
                        # Make sure you set the dim = 1, so it will actually look for the top probability
                        # along the columns
                        top_ps, top_class = ps.topk(1, dim=1)

                        # Check for equality against your labels with the equality tensor
                        equality = top_class == labels.view(*top_class.shape)

                        # Update accuracy, remember to use equality, once you change it to FloatTensor
                        # than you can use torch.mean
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                    # Print Information
                    print('Epoch: {}/{}'.format(epoch + 1, epochs),
                          "Training Loss: {:.3f}...".format(running_loss / print_everything),
                          "Validation Loss: {:.3f}...".format(valid_loss / len(validloaders)),
                          'Validation Accuracy {:.3f}%'.format(accuracy / len(validloaders)))

                    # Set running loss back to 0
                    runninng_loss = 0

                    # Set model back to Training
                    model.train()
    print("Training Completed")


def get_transformation(train=True):
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

    if train:
        # Add data augmentation transformations for training
        train_transforms = [transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224)]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        # Return only the common transformations for validation/testing
        valid_test_transformations = [transforms.Resize(256),
                                     transforms.CenterCrop(224)] # Center crop to 224x224]
        return transforms.Compose(valid_test_transformations + base_transforms)


def get_datasets(file_path):

    train_dir, valid_dir, test_dir = determine_dataset(file_path)

    train_datasets = datasets.ImageFolder(train_dir, transform=get_transformation(False))
    valid_datasets = datasets.ImageFolder(valid_dir, transform=get_transformation(True))
    test_datasets = datasets.ImageFolder(test_dir, transform=get_transformation(True))

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=get_transformation(False)),
        'validation': datasets.ImageFolder(valid_dir, transform=get_transformation(True)),
        'test': datasets.ImageFolder(test_dir, transform=get_transformation(True))
    }

    return train_datasets, valid_datasets, test_datasets, image_datasets


def get_dataloaders(train_datasets, valid_datasets, test_datasets):
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
    return train_loader, valid_loader, test_loader


def load_cat_to_name(file_path):
    with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name


def setup_hyper_params(model_name, hidden_units):
    model = get_model_architecture(model_name)

    for param in model.parameters():
        param.requires_grad = False

    input_size = 0

    if model_name == 'vgg16':
        input_size = model.classifier[0].in_features
    elif model_name == 'alexnet':
        input_size = model.classifier[1].in_features
    elif model_name == 'densenet121':
        input_size = model.classifier.in_features

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
    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.001)

    # move the model to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set up paramter that are used to train our model
    epochs = 1
    running_loss = 0
    steps = 0
    print_everything = 10



def main():
    input_args = get_input_args()
    train_datasets, valid_datasets, test_datasets, image_datasets = get_datasets(input_args.dir)
    train_loader, valid_loader, test_loader = get_dataloaders(train_datasets, valid_datasets, test_datasets)

    model_name = input_args.arch
    hidden_units = input_args.hidden_unit
    epochs = input_args.epochs
    learning_rate = input_args.learning_rate
    device = torch.device(input_args.cuda if torch.cuda.is_available() else 'cpu')
    model = get_model_architecture(model_name)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()









