import sys
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

argparser = ArgumentParser()
argparser.add_argument('--device', type=int, required=True, 
        help='id of device to run training on.')
argparser.add_argument('--dir', type=str, 
        help='directory where model is located to use for inference. \
                segmentation maps are written here as well in inference/.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')


args = argparser.parse_args()
os.makedirs(os.path.join([args.dir, '/inference/logs/']), exist_ok=True)

device      = torch.device(f'cuda:{args.device}')

'''
Load the model and its parameters.
'''

'''
Load the dataset and prepare dataloader.
'''

'''
Iterate over the dataset using \hat{P}(Y|X) to crete segmentation maps.
The 3d network naturally outputs the 3d segmentation maps but 2d will re-
quire some work. In both cases we must make sure the necessary metadata
has been stored in order to restore the output space to its correct form.
'''

'''
2d 
'''










'''
3d 
'''


