import argparse
import os
import errno

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # command line options
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'Path to the folder of the flower images')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Path to save the model checkpoints')
    parser.add_argument('--learn_rate', type = str, default = '0.001', help = 'Model learning rate')  
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'Model architecture (densenet121, vgg16)')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Hidden units')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Number of epochs')
    parser.add_argument('--to_device', type = str, default = 'gpu', help = 'Run model on GPU (default)')
    return parser.parse_args()


def make_folder(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise