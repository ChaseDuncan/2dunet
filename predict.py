from argparse import ArgumentParser
from glob import glob
import sys 
import os
import random

'''
    set up variables
'''
#   invariant data dirs
pred_dir = 'brats2020/2dunet/model-predictions'
inpt_dir = 'brats2020/2dunet/test-preprocessed-nocrop/data/'
#inpt_dir = 'brats2020/2dunet/train-preprocessed-v2/data/'

#   model params
input_nc    = 4
output_nc   = 3
num_downs   = 3
ngf         = 64

#   CLI argument parsing
argparser = ArgumentParser()

argparser.add_argument('--dir', type=str, required=True,
        help='directory for all model output, logs, checkpoints, etc.')
argparser.add_argument('--device', type=int, required=True, help='id of device to run training on.')
argparser.add_argument('--batch_size', type=int, default=150, help='batch size. (default: 150)')
argparser.add_argument('--seed', type=int, required=True, help='random seed to use for training.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')
argparser.add_argument('--freeze_encoder', action='store_true', 
        help='train UNet with frozen encoder. (default: False).')

args            = argparser.parse_args()
seed            = args.seed
try:
    model_name  = args.dir.split('/')[-1]
except:
    print(f'invalid model directory: {args.dir}')
output_dir      = os.path.join(pred_dir, model_name)
os.makedirs(output_dir, exist_ok=True)
checkpoint = sorted(glob(f'{args.dir}/checkpoints/*'))[-1]


'''
    prediction
'''
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from dataloader import BraTS20202dPredict
from models import UNetGenerator

from utils import *


#   sundry
device      = torch.device(f'cuda:{args.device}')

#   fix all randomness for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_deterministic(True)

if args.debug:
    inpt_dir    = 'brats2020/2dunet/debug/'
    num_workers =0
else:
    num_workers = 4

def collate_fn(batch):
    inputs = []
    samples = []
    for src, _, sample in batch:
        inputs.append(src)
        samples.append(sample)
    return torch.cat(inputs), samples

pred_dataset                     = BraTS20202dPredict(inpt_dir)
# no batching so it's not
pred_loader                 = DataLoader(pred_dataset, shuffle=False, #collate_fn=collate_fn,
                                num_workers=num_workers, batch_size=args.batch_size, pin_memory=False)

#   model setup
model           = UNetGenerator(input_nc, output_nc, num_downs, ngf=ngf, 
                                freeze_encoder=args.freeze_encoder)

checkpoint = torch.load(checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)

def store_outputs(sigs, filenames):
    for output, filename in zip(sigs, filenames):
        sample                                          = filename#[0]
        patient                                         = sample.split('.')[0]
        patient_output_dir                              = os.path.join(output_dir, patient)

        os.makedirs(patient_output_dir, exist_ok=True)
        np.save(os.path.join(patient_output_dir, sample), sigs)

#   simple prediction loop: load it, predict X, save it.
with torch.no_grad():
    model.eval()
    for j, (src, filenames) in enumerate(tqdm(pred_loader)):
        #   default collate_fn returns patient string in a tuple

        src_pad                                         = torch.zeros((args.batch_size, 4, 240, 240))
        src_pad[..., :src.shape[-2], :src.shape[-1]]    = src
        src_pad                                         = src_pad.to(device).float()

        output                                          = model(src_pad) 
        sigs                                            = \
                torch.sigmoid(output).cpu().numpy()[..., :src.shape[-2], :src.shape[-1]]
        store_outputs(sigs, filenames)

## breaks if batch_size > 1
#stuff['avg_eval_loss']  = avg_eval_loss 
#log_epoch(stuff)
#    
#    if epoch % args.save_freq == 0:
#        torch.save({
#            'epoch': epoch,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'avg_train_loss': avg_train_loss,
#            'avg_eval_loss': avg_eval_loss,
#            }, f'{args.dir}/checkpoints/epoch_{epoch:03}.pt'
#            )

