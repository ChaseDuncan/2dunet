from argparse import ArgumentParser

import tabulate
argparser = ArgumentParser()
argparser.add_argument('--device', type=int, required=True, help='id of device to run training on.')
argparser.add_argument('--seed', type=int, required=True, help='random seed to use for training.')
argparser.add_argument('--dir', type=str, help='directory for all model output, logs, checkpoints, etc.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')
argparser.add_argument('--three_d', action='store_true', 
        help='train a 3d model. (default: False)')
argparser.add_argument('--freeze_encoder', action='store_true', 
        help='train UNet with frozen encoder. (default: False).')
argparser.add_argument('--save_freq', type=int, default=1, 
        help='epoch frequency with which to checkpoint. (default = 1)')

args = argparser.parse_args()

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

from dataloader import BraTS20202d, BraTS20203d
from models import UNetGenerator, UNetGenerator3d
from losses import BraTSBCEWithLogitsLoss
from sklearn.model_selection import train_test_split 

from utils import *

os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir + '/logs/', exist_ok=True)
os.makedirs(args.dir + '/checkpoints/', exist_ok=True)

#   sundry
seed        = args.seed
device      = torch.device(f'cuda:{args.device}')

#   model params
input_nc    = 4
output_nc   = 3
num_downs   = 3
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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

tr_data_dir    = 'brats2020/2dunet/train-preprocessed-v2/data/'
te_data_dir    = 'brats2020/2dunet/test-preprocessed-v2/data/'


if args.debug:
    tr_data_dir    = 'brats2020/2dunet/debug/'
    te_data_dir = 'brats2020/2dunet/debug/'
    num_workers =0
    batch_size = 1
elif args.three_d:
    tr_data_dir = 'brats2020/2dunet/train/'
    te_data_dir = 'brats2020/2dunet/test/'
    batch_size = 5
    num_workers = 4
else:
    num_workers = 4
    batch_size = 150

#   data setup
print(f'loading training data from: {tr_data_dir}')
print(f'loading test data from: {te_data_dir}')

if args.three_d:
    train_dataset               = BraTS20203d(tr_data_dir)
    test_dataset                = BraTS20203d(te_data_dir)
else:
    train_dataset               = BraTS20202d(tr_data_dir, only_tumor=True)
    test_dataset                = BraTS20202d(te_data_dir, only_tumor=True)

train_loader                = DataLoader(train_dataset, shuffle=True, 
                                    num_workers=num_workers, batch_size=batch_size)
                                        #collate_fn=collate_fn)
# batch_size == 1 required for the image logging to work properly
test_loader                 = DataLoader(test_dataset, shuffle=False, 
                                num_workers=num_workers, batch_size=1)

#   model setup
if args.three_d:
    model           = UNetGenerator3d(input_nc, output_nc, num_downs, ngf=ngf, 
                                    freeze_encoder=args.freeze_encoder).to(device)
    loss            = BraTSBCEWithLogitsLoss()
    optimizer       = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
else:
    model           = UNetGenerator(input_nc, output_nc, num_downs, ngf=ngf, 
                                    freeze_encoder=args.freeze_encoder).to(device)
    loss            = BraTSBCEWithLogitsLoss()
    optimizer       = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

start_epoch = 0
writer          = SummaryWriter(log_dir=f'{args.dir}/logs/{start_epoch}/')
table_out_columns = ['ep', 'train loss', 'train_dice_et', 'train_dice_wt','train_dice_tc',\
                'test loss', 'test_dice_et', 'test_dice_wt','test_dice_tc']


def log_epoch(stuff):
    ''' convenience function for logging everything. keeps the main loop clean.'''
    epoch   = stuff['epoch']
    writer.add_scalar('Loss/train', stuff['train_loss'], epoch)
    writer.add_scalar('Dice/enhancing train', stuff['train_dice'][0], epoch)
    writer.add_scalar('Dice/whole train', stuff['train_dice'][1], epoch)
    writer.add_scalar('Dice/core train', stuff['train_dice'][2], epoch)

    writer.add_scalar('Loss/test', stuff['test_loss'], epoch)
    writer.add_scalar('Dice/enhancing test', stuff['test_dice'][0], epoch)
    writer.add_scalar('Dice/whole test', stuff['test_dice'][1], epoch)
    writer.add_scalar('Dice/core test', stuff['test_dice'][2], epoch)

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
    if not pred.is_contiguous():
        pred = pred.contiguous()
    if not trgt.is_contiguous():
        trgt = trgt.contiguous()
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

for epoch in range(epochs):
    model.train()
    stuff   = {'epoch': epoch, 'images': []}
    test_loss = train_loss  = 0

    for i, (src, tgt) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        src, tgt    = src.to(device).float(), tgt.to(device).float()
        output      = model(src) 

        print(f'outputsize: {output.size()}, tgtsize {tgt.size()}')
        loss_e      = loss(output, tgt)
        
        train_loss    = loss_e/(i+1) + (i/(i+1))*train_loss
        loss_e.backward()
        optimizer.step()

    stuff['train_loss'] = train_loss 
    with torch.no_grad():
        model.eval()
        train_loss   = 0
        stuff['train_dice'] = np.zeros(3)
        for j, (src, tgt) in enumerate(tqdm(train_loader)):
            src, tgt    = src.to(device).float(), tgt.to(device).float()
            output          = model(src) 
            loss_e          = loss(output, tgt)
            # updata average loss
            stuff['train_dice']        = compute_dice(output, tgt)/(j+1) + (j/(j+1))*stuff['train_dice']
            train_loss   = loss_e/(j+1) + (j/(j+1))*train_loss

    stuff['test_loss'] = test_loss 
    with torch.no_grad():
        model.eval()
        test_loss   = 0
        stuff['test_dice'] = np.zeros(3)
        for j, (src, tgt) in enumerate(tqdm(test_loader)):
            src, tgt    = src.to(device).float(), tgt.to(device).float()
            output          = model(src) 
            loss_e          = loss(output, tgt)
            # updata average loss
            stuff['test_dice']        = compute_dice(output, tgt)/(j+1) + (j/(j+1))*stuff['test_dice']
            test_loss   = loss_e/(j+1) + (j/(j+1))*test_loss
            if j+1 in examples_to_track:
                sigs  = torch.sigmoid(output) 
                sigs_npy = sigs.cpu().numpy()
                preds  = torch.zeros(sigs.size(), dtype=torch.uint8)
                preds[torch.where(sigs > 0.5)] = 255
                tgt_scaled                  = tgt*255

                if len(preds[1].shape) == 3:
                    preds = preds[..., 100].squeeze()
                img = torch.cat([preds, tgt_scaled.cpu()], dim=1).to(torch.uint8)
                stuff['images'].append((j+1, img))

    train_stats = [ epoch + 1, stuff['train_loss']]+ stuff['train_dice'].tolist() \
                    + [stuff['test_loss']] + stuff['test_dice'].tolist()

    table_train = tabulate.tabulate([train_stats], table_out_columns, tablefmt="simple", floatfmt="8.4f")
    print(table_train)

    # breaks if batch_size > 1
    stuff['test_loss']  = test_loss
    log_epoch(stuff)
    
    if epoch % args.save_freq == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_dice': stuff['train_dice'],
            'test_dice': stuff['test_dice']
            }, f'{args.dir}/checkpoints/epoch_{epoch:03}.pt'
            )

