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

from models import UNetGenerator
from dataloader import BraTS2020Test2d

argparser = ArgumentParser()
argparser.add_argument('--device', type=int, required=True, 
        help='id of device to run training on.')
argparser.add_argument('--dir', type=str, 
        help='directory where model is located to use for inference. \
                segmentation maps are written here as well in inference/.')
argparser.add_argument('--data_dir', type=str, 
        help='path to directory of data to annotate.')
argparser.add_argument('--debug', action='store_true', 
        help='use debug mode which only uses one example to train and eval.')

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

checkpoint  = torch.load('data/ensemble/0/checkpoints/epoch_012.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model       = model.to(device)

'''
Load the dataset and prepare dataloader.
'''
dataset     = BraTS2020Test2d(args.data_dir, num_downs)
# batch size must be one
dataloader  = DataLoader(dataset, collate_fn=lambda x: x[0])

def create_and_save_segmentation(total_seg, d):
    preds   = np.zeros(total_seg.shape) 
    preds[np.where(total_seg > 0.80)] = 1
    for i in range(3):
        img = nib.Nifti1Image(preds[i], d['affine'], header=d['header'])
        nib.save(img, f'deleteme{i}.nii.gz')

with torch.no_grad():
    model.eval()
    for i, d in enumerate(tqdm(dataloader)):
        # label the slices
        total_seg   = []
        j=0
        for slc, nnz, orig_shape in d['sagittal_slices']:
            slc_c = slc.copy()
            slc         = torch.from_numpy(slc).unsqueeze(dim=0).to(device, dtype=torch.float)
            lgt         = model(slc[:, :-1]) 
            preds       = torch.sigmoid(lgt).cpu()
            
            if j == 100:
                sigs                        = torch.sigmoid(lgt) 
                preds_test                       = torch.zeros(sigs.size(), dtype=torch.uint8)
                preds_test[torch.where(sigs > 0.5)] = 255
                np.save('preds', preds_test)
                np.save('slc_c', slc_c)
                np.save('slc', slc.cpu())

            preds       = preds[..., :(nnz[0][1]-nnz[0][0]+1), :(nnz[1][1]-nnz[1][0]+1)].squeeze(0)
            j+=1
            pred_slc    = np.zeros([3] + [*orig_shape])
            pred_slc[..., nnz[0][0]:nnz[0][1]+1, nnz[1][0]:nnz[1][1]+1]   = preds
            total_seg.append(pred_slc)
        total_seg = np.stack(total_seg, axis=1)
        np.save('total_seg', total_seg)
        create_and_save_segmentation(total_seg, d)
        break
