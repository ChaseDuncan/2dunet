from glob import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *

class BraTS20202d(Dataset):
    def __init__(self, data_dir, only_tumor=False):
        ''' @param: data_dir (str)      : path to directory containing brats 2018/2019/2020
                                          data that is processed according the preprocessing.py.
            @param: patch_size (tuple)  : the size of the patch to use for training.
        '''
        self.filenames  = glob(f'{data_dir}/*.npy')
        if only_tumor:
            self.filenames = [f for f in self.filenames if 'tum' in f]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        patient = self.filenames[idx]
        imgs_npy = np.load(patient)
        found_good_sample = False
        while(not found_good_sample):
            try:
                crop, _, _ = crop_to_brain(imgs_npy)
                found_good_sample=True
            except:
                print(f'{self.filenames.pop(idx)} cannot be parsed.')
        X, Y = torch.from_numpy(crop[:4]), torch.from_numpy(crop[4:])
        return X, Y 
        #return torch.from_numpy(imgs_npy).squeeze()
