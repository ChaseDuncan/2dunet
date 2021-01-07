import sys
import os
import random

import torch
import torchvision
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
argparser.add_argument('--dir', type=str, help='directory for all model output, logs, checkpoints, etc.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')
argparser.add_argument('--freeze_encoder', action='store_true', 
        help='train UNet with frozen encoder. (default: False).')
argparser.add_argument('--save_freq', type=int, default=1, 
        help='epoch frequency with which to checkpoint. (default = 1)')

args = argparser.parse_args()

os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir + '/logs/', exist_ok=True)
os.makedirs(args.dir + '/checkpoints/', exist_ok=True)

#   sundry
seed        = 1234
device      = torch.device(f'cuda:{args.device}')

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

if args.debug:
    data_dir    = 'brats2020/2d-preprocessed-debug/data'
else:
    data_dir    = 'brats2020/2d-preprocessed/data'

#   data setup

print(f'loading data from: {data_dir}')
dataset                     = BraTS20202d(data_dir)
indices                     = np.arange(len(dataset))
train_indices, test_indices = train_test_split(indices, train_size=0.8)
with open(args.dir + '/crossval_idxs.txt', "w") as f:
    f.write(f'train_indices\t{train_indices}\ntest_indices\t{test_indices}')

# Warp into Subsets and DataLoaders
train_dataset               = Subset(dataset, train_indices)
test_dataset                = Subset(dataset, test_indices)

train_loader                = DataLoader(train_dataset, shuffle=True, num_workers=16, batch_size=150)
# batch_size > 1 required for the image logging to work properly
test_loader                 = DataLoader(test_dataset, shuffle=False, num_workers=16, batch_size=1)

#   model setup
model           = UNetGenerator(input_nc, output_nc, num_downs, ngf=ngf, 
                                freeze_encoder=args.freeze_encoder).to(device)
loss            = BraTSBCEWithLogitsLoss()
optimizer       = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

start_epoch = 0
writer          = SummaryWriter(log_dir=f'{args.dir}/logs/{start_epoch}/')

def log_epoch(stuff):
    ''' convenience function for logging everything. keeps the main loop clean.'''
    epoch   = stuff['epoch']
    writer.add_scalar('Loss/train', stuff['avg_train_loss'], epoch)
    writer.add_scalar('Loss/eval', stuff['avg_eval_loss'], epoch)
    writer.add_scalar('Dice/enhancing', stuff['avg_dice'][0], epoch)
    writer.add_scalar('Dice/whole', stuff['avg_dice'][1], epoch)
    writer.add_scalar('Dice/core', stuff['avg_dice'][2], epoch)
    for img in stuff['images']:
        grid    = torchvision.utils.make_grid(
                    [s.squeeze() for s in torch.split(img[1].squeeze(), 1)], nrow=1, pad_value=255)
        # need a dummy 'channel' dimension for the tensorboard api
        grid    = grid.unsqueeze(1)
        writer.add_images(f'Images/{img[0]}', grid, global_step=epoch)
    # TODO: Haus, NLL, images
    #writer.add_scalar('Dice/whole', np.random.random(), n_iter)
    #writer.add_scalar('Dice/enhancing', np.random.random(), n_iter)
    #writer.add_scalar('Dice/core', np.random.random(), n_iter)
    
def dice_coeff(pred, trgt):
    smooth  = 1.

    pflat   = pred.view(-1)
    tflat   = trgt.view(-1)

    intsc   = (pflat*tflat).sum()

    return (2*intsc + smooth)/ (pflat.sum() + tflat.sum() + smooth)

def compute_dice(output, target):
    sigs                            = torch.sigmoid(output) 
    preds                           = torch.zeros(sigs.size(), dtype=torch.uint8)
    preds[torch.where(sigs > 0.5)]  = 1
    preds                           = preds.cuda(output.get_device())
    dice = np.zeros(3)
    for i in range(3):
        dice[i] = dice_coeff(preds[:, i].squeeze(), target[:, i].squeeze())
    return dice

examples_to_track   = [random.randint(0, len(test_dataset)) for _ in range(10)]
print(examples_to_track)
for epoch in range(epochs):
    model.train()
    stuff   = {'epoch': epoch, 'images': []}
    # deleteme

    avg_train_loss    = 0
    for i, (src, tgt) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        src, tgt    = src.to(device), tgt.to(device)
        output      = model(src) 
        loss_e      = loss(output, tgt)
        
        avg_train_loss    = loss_e/(i+1) + (i/(i+1))*avg_train_loss
        loss_e.backward()
        optimizer.step()

    stuff['avg_train_loss'] = avg_train_loss 
    with torch.no_grad():
        model.eval()
        avg_eval_loss   = 0
        stuff['avg_dice'] = np.zeros(3)
        for j, (src, tgt) in enumerate(tqdm(test_loader)):
            src, tgt        = src.to(device), tgt.to(device)
            output          = model(src) 
            loss_e          = loss(output, tgt)
            # updata average loss
            stuff['avg_dice']        = compute_dice(output, tgt)/(j+1) + (j/(j+1))*stuff['avg_dice']
            avg_eval_loss   = loss_e/(j+1) + (j/(j+1))*avg_eval_loss
            if j+1 in examples_to_track:
                sigs                        = torch.sigmoid(output) 
                preds                       = torch.zeros(sigs.size(), dtype=torch.uint8)
                preds[torch.where(sigs > 0.5)] = 255
                tgt_scaled                  = tgt*255
                img                         = torch.cat([preds, tgt_scaled.cpu()], dim=1).to(torch.uint8)
                np.save(f'deleteme{j+1}', img.cpu().numpy())
                stuff['images'].append((j+1, img))

    # breaks if batch_size > 1
    stuff['avg_eval_loss']  = avg_eval_loss 
    log_epoch(stuff)
    
    if epoch % args.save_freq == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_train_loss': avg_train_loss,
            'avg_eval_loss': avg_eval_loss,
            }, f'{args.dir}/checkpoints/epoch_{epoch:03}.pt'
            )

