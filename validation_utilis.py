import os
import pathlib

import torch

from config import DATA_ROOT

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
