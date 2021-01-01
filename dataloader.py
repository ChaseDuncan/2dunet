import tables

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BraTS20202d(Dataset):
    def __init__(self, data_file, patch_size=(200, 200)):
        self.hdf5_file     = tables.open_file(data_file, mode='r+')
        self.data_storage  = self.hdf5_file.root.data    
        self.meta_storage  = self.hdf5_file.root.meta

    def __len__(self):
        return len(self.data_storage)

    def __getitem__(self, idx):
        # remove, for debugging
        #idx = 54321
        return torch.from_numpy(self.data_storage[idx][:4, :128, :128]),\
                            torch.from_numpy(self.data_storage[idx][4:, :128, :128])
        #return torch.from_numpy(self.data_storage[idx][:4]), torch.from_numpy(self.data_storage[idx][4:])

