from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BraTS20202d(Dataset):
    def __init__(self, data_dir, patch_size=(200, 200)):
        ''' @param: data_dir (str)      : path to directory containing brats 2018/2019/2020
                                          data that is processed according the preprocessing.py.
            @param: patch_size (tuple)  : the size of the patch to use for training.
        '''
        self.filenames  = glob(f'{data_dir}/*.npy')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, patch_size=128):
        # remove, for debugging
        #idx = 54321
        data                        = np.load(self.filenames[idx], mmap_mode='r')
        data_dim                    = len(data.shape) - 1
        diff                        = np.array([patch_size]*data_dim) - data.shape[1:]
        pads                        = np.zeros(data_dim)
        if len(np.where(diff > 0)[0]) > 0:
            pads = np.where(diff > 0, diff, pads)
        pad_width                   = [(0, 0)] + [(0, int(p)) for p in pads] 
        data                        = np.pad(data, pad_width)
        return torch.from_numpy(data[:4, :patch_size, :patch_size]).float(),\
                            torch.from_numpy(data[4:, :patch_size, :patch_size]).float()
        #return torch.from_numpy(self.data_storage[idx][:4]), torch.from_numpy(self.data_storage[idx][4:])

