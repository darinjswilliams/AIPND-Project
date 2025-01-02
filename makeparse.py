import argparse
import pathlib

from config import DATA_ROOT

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
