import sys
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import nibabel as nib

from glob import glob
from models import UNetGenerator
from dataloader import BraTS2020Test2d, BraTS2020Training2d

argparser = ArgumentParser()
argparser.add_argument('--device', type=int, required=True, 
        help='id of device to run training on.')
argparser.add_argument('--dir', type=str, 
        help='directory where model is located to use for inference. \
                segmentation maps are written here as well in inference/.')
argparser.add_argument('--model', type=str, 
        help='path to model is located to use for inference. \
                segmentation maps are written here as well in inference/.')
argparser.add_argument('--data_dir', type=str, 
        help='path to directory of data to annotate.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')
argparser.add_argument('--twod', action='store_true', 
        help='segment 2d data')
argparser.add_argument('--threed', action='store_true', 
        help='segment 3d data')

args = argparser.parse_args()
os.makedirs(os.path.join(args.dir, 'inference/logs/'), exist_ok=True)

device      = torch.device(f'cuda:{args.device}')

'''
Load the model and its parameters.
'''
#   model hyperparams
input_nc    = 4
output_nc   = 3
num_downs   = 3
ngf         = 64
model       = UNetGenerator(input_nc, output_nc, num_downs, ngf=ngf)

checkpoint  = torch.load(args.model)
model.load_state_dict(checkpoint['model_state_dict'])
model       = model.to(device)

#os.makedirs('deletemedir', exist_ok=True)
'''
2d/trainingdata
'''
#if args.twod:
#    dataset     = BraTS2020Training2d(args.data_dir)
#    # batch size must be one
#    dataloader  = DataLoader(dataset)
#    preds = []
#    with torch.no_grad():
#        model.eval()
#        for i, ((src, tgt, nonzero, orig_shape), case_name) in enumerate(tqdm(dataloader)):
#            src     = src.to(device).float()
#            output  = model(src)
#            sigs    = torch.sigmoid(output) 
#            preds   = torch.zeros(sigs.size(), dtype=torch.uint8)
#            preds[torch.where(sigs > 0.5)]  = 1
#            pid     = case_name[0].split('/')[-1]
#            np.save(os.path.join('deletemedir', pid), preds.cpu())

'''
3d/trainingdata
'''
#if args.threed:
#    dataset     = BraTS2020Test2d(args.data_dir, num_downs)
#    dataloader  = DataLoader(dataset, collate_fn=lambda x: x[0])
#    preds       = []
#    with torch.no_grad():
#        model.eval()
#        for i, d_dict in enumerate(tqdm(dataloader)):
#            for j, slc in enumerate(d_dict['coronal_slices']):
#                slc         = slc[0]
#                slc, tgt    = slc[:-1], slc[-1]
#                slc = slc.unsqueeze(0).to(device).float()
#                slc =model(slc)
#                probs       = torch.sigmoid(slc).cpu()
#                pred        = np.zeros(slc.shape)
#                pred[np.where(probs>0.5)]  = 1
#                # TODO: remove after debug
#                preds.append(pred[0][1])
#
#            pred = np.array(preds)
#            img = nib.Nifti1Image(pred, affine=np.eye(4))
#            nib.save(img, 'deleteme.nii.gz')
#            
#
#            break

segs    = glob('deletemedir/*_094*')

for seg in segs:
    seg_mat = np.load(seg).squeeze()
 
