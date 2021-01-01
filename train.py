import sys
import os
import argparse
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from dataloader import BraTS20202d
from models import UNetGenerator
from losses import BraTSBCEWithLogitsLoss
from sklearn.model_selection import train_test_split 

from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--device', type=int, required=True, help='id of device to run training on.')
argparser.add_argument('--model_dir', type=str, help='directory for all model output, logs, checkpoints, etc.')
args = argparser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.model_dir + '/logs/', exist_ok=True)
os.makedirs(args.model_dir + '/checkpoints/', exist_ok=True)

#   sundry
seed        = 1234
device      = torch.device(f'cuda:{args.device}')
debug       = False
debug       = True

#   model params
input_nc    = 4
output_nc   = 3
num_downs   = 4
ngf         = 64
#   optim params
lr          = 8e-3
momentum    = 0.9
wd          = 1e-4
#   train params
epochs      = 150
#   fix all randomness for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_deterministic(True)

if debug:
    hdf5_file   = 'brats2020/2dhdf5/test.data.hdf5'
else:
    hdf5_file   = 'brats2020/2dhdf5/data.hdf5'

#   data setup

print(f'loading: {hdf5_file}')
dataset                     = BraTS20202d(hdf5_file)
indices                     = np.arange(len(dataset))
train_indices, test_indices = train_test_split(indices, train_size=0.8)
with open(args.model_dir + '/crossval_idxs.txt', "w") as f:
    f.write(f'train_indices\t{train_indices}\ntest_indices\t{test_indices}')
# Warp into Subsets and DataLoaders
train_dataset               = Subset(dataset, train_indices)
test_dataset                = Subset(dataset, test_indices)

train_loader                = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=100)
test_loader                 = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=100)

#   model setup
model           = UNetGenerator(input_nc, output_nc, num_downs, ngf=ngf).to(device)
loss            = BraTSBCEWithLogitsLoss()
optimizer       = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

for epoch in range(epochs):
    model.train()
    # deleteme
    i=0
    for src, tgt in tqdm(train_loader):
        optimizer.zero_grad()
        src, tgt    = src.to(device), tgt.to(device)
        output      = model(src) 
        loss_e      = loss(output, tgt)
        loss_e.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        avg_loss    = 0
        i           = 0
        for src, tgt in tqdm(test_loader):
            src, tgt    = src.to(device), tgt.to(device)
            output      = model(src) 
            loss_e      = loss(output, tgt)
            avg_loss    = loss_e/(i+1) + (i/(i+1))*avg_loss
            i+=1

    print(f'epoch\t{epoch:3}.\t average loss on eval\t{avg_loss:3f}.')


