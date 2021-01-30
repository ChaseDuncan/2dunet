from glob import glob
import pickle
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
        self.filenames          = glob(f'{data_dir}/*.npy')

        if only_tumor:
            self.filenames      = [f for f in self.filenames if 'tum' in f]

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

class BraTS20202dPredict(BraTS20202d):
    def __init__(self, data_dir):
        super().__init__(os.path.join(data_dir, 'data'))
        self.meta = glob(os.path.join(data_dir, 'meta'))
        self.meta = sorted(self.meta)
        self.data = sorted(self.data)

    def __getitem__(self, idx):
        patient = self.filenames[idx]
        imgs_npy = np.load(patient)
        X, Y = torch.from_numpy(imgs_npy[:4]), torch.from_numpy(imgs_npy[4:])
        #return X.squeeze().unsqueeze(0), Y, ".".join(patient.split('.')[:-1]).split('/')[-1]
        return X.squeeze().unsqueeze(0), ".".join(patient.split('.')[:-1]).split('/')[-1]


class BraTS20203d(Dataset):
    def __init__(self, data_dir):
        self.list_of_lists_of_filenames = glob(f'{data_dir}/*.nii.gz')
        self.list_of_lists_of_filenames = [
                sorted([ f for f in self.list_of_lists_of_filenames if "t1.nii.gz" in f ]),
                sorted([ f for f in self.list_of_lists_of_filenames if "t1ce.nii.gz" in f ]),
                sorted([ f for f in self.list_of_lists_of_filenames if "t2.nii.gz" in f ]),
                sorted([ f for f in self.list_of_lists_of_filenames if "flair.nii.gz" in f ]),
                sorted([ f for f in self.list_of_lists_of_filenames if "seg.nii.gz" in f ])
        ]

    def __len__(self):
        return len(self.list_of_lists_of_filenames[0])

    def _load_images(self, idx):
        images = []
        for m in self.list_of_lists_of_filenames:
            image = nib.load(m[idx])
            images.append(image)
            header = image.header
        images = np.array([image.get_fdata() for image in images])
        return images, header

    
    def __getitem__(self, idx):
        case, header = self._load_images(idx)
        brain_crop, nonzero, orig_shape = read_brain(case, no_crop=False)
        return torch.from_numpy(brain_crop[:4]), torch.from_numpy(brain_crop[4:])
