from glob import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
''' drug over from preprocessing where it was no longer needed '''
def dim_diff(self, dim, num_downs):
    mult    = 2**num_downs*np.ceil(dim / 2**num_downs)
    return int(mult - dim)


class BraTS20202d(Dataset):

    def nonzero_coords(self, mode): 
        ''' gives the nonzero coordinates of a particular mode'''
        nonzero = np.where(mode!= 0) 
        if len(nonzero) == 2:
            nonzero_c = [[np.min(nonzero[0]), np.max(nonzero[0])], 
                [np.min(nonzero[1]), np.max(nonzero[1])]]

        elif len(nonzero) == 3:
            nonzero_c = [[np.min(nonzero[0]), np.max(nonzero[0])], 
                [np.min(nonzero[1]), np.max(nonzero[1])], 
                [np.min(nonzero[2]), np.max(nonzero[2])]]
        else:
            raise IndexError(f'Invalid data dimension: {len(nonzero)}')
        return nonzero_c

    def normalize_case(self, imgs_npy):
        # now we create a brain mask that we use for normalization
        nonzero_masks = [i != 0 for i in imgs_npy]
        brain_mask = np.zeros(imgs_npy.shape[1:5], dtype=bool)
        
        for i in range(len(nonzero_masks)):
            brain_mask = brain_mask | nonzero_masks[i]

        # now normalize each modality with its mean and standard deviation (computed within the brain mask)
        for i in range(len(imgs_npy) - 3):
            mean = imgs_npy[i][brain_mask].mean()
            std = imgs_npy[i][brain_mask].std()
            imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
            imgs_npy[i][brain_mask == 0] = 0
        return imgs_npy
    
    def dim_diff(self, dim, num_downs):
        mult    = 2**num_downs*np.ceil(dim / 2**num_downs)
        return int(mult - dim)

    def crop_to_brain(self, case, num_downs=4):
        # crop to t1 for no particular reason
        nonzero     = self.nonzero_coords(case[1])
        orig_shape  = case.shape[1:]
        if len(case.shape) == 4:
            brain_crop  = case[:, nonzero[0][0]:nonzero[0][1],
                    nonzero[1][0]:nonzero[1][1],
                    nonzero[2][0]:nonzero[2][1]]
        elif len(case.shape) == 3:
            brain_crop  = case[:, nonzero[0][0]:nonzero[0][1]+1,
                    nonzero[1][0]:nonzero[1][1]+1]
            # make sure image is big enough to be pushed through the network
            brain_crop  = np.pad(brain_crop, 
                    ( # pad each dimension so that it is a multiple of 2^num_downs
                        (0,0),
                         (0, 220-brain_crop.shape[1]), 
                         (0, 220-brain_crop.shape[2]))
                       # (0, self.dim_diff(brain_crop.shape[1], num_downs)), 
                       # (0, self.dim_diff(brain_crop.shape[2], num_downs))
                    )
        else:
            raise IndexError(f'Invalid data dimension: {len(case.shape)}')
        return brain_crop, nonzero, orig_shape

    def crop_to_brain_and_normalize(self, case, num_downs): 
        brain_crop, nonzero, orig_shape = self.crop_to_brain(case, num_downs=num_downs)
        return torch.from_numpy(self.normalize_case(brain_crop)), nonzero, orig_shape

class BraTS2020Training2d(BraTS20202d):
    def __init__(self, data_dir):
        ''' @param: data_dir (str)      : path to directory containing brats 2018/2019/2020
                                          data that is processed according the preprocessing.py.
            @param: patch_size (tuple)  : the size of the patch to use for training.
        '''
        self.filenames  = glob(f'{data_dir}/*.npy')

    def __len__(self):
        return len(self.filenames)

    def read_brain(self, case, num_downs=3):
        ''' From the perspective of the model, a slice is an example, not the whole brain.
        Therefore for inference we unpack the dataset into slices along each axis, crop,
        normalize, and store for inference along with the metadata needed to reconstruct
        the probability map in the problem space.
        ''' 
        img         = np.load(case)
        brain_crop, nonzero, orig_shape = self.crop_to_brain_and_normalize(img, num_downs)
        return brain_crop[:4], brain_crop[4:], nonzero, orig_shape

    def __getitem__(self, idx):
        return self.read_brain(self.filenames[idx]), self.filenames[idx]

class BraTS2020Test2d(BraTS20202d):
    '''
        I decided to have the dataset directly consume the source test data rather than
        having an interim preprocessing script as in training. This is for a few reasons
        the dataset is relatively small, we're only doing a single forward pass, I don't
        expect to do it often. Given all that, it seemed messy to split this functionality
        across multiple files.

        As a result, this is a pretty densely featured Dataset subclass.
    '''
    def __init__(self, test_data_dir, num_downs):
        self.filenames       = glob(f'{test_data_dir}/*/*.nii.gz')
        if len(self.filenames) == 0:
            self.filenames       = glob(f'{test_data_dir}/*.nii.gz')

        self.filenames       = [[f for f in self.filenames if 't1.' in f],
                        [f for f in self.filenames if 't1ce.' in f],
                        [f for f in self.filenames if 't2.' in f],
                        [f for f in self.filenames if 'flair.' in f],
                        [f for f in self.filenames if 'seg.' in f]

                        ]
        self.num_downs       = num_downs


    def read_brain(self, case, num_downs=3):
        ''' From the perspective of the model, a slice is an example, not the whole brain.
        Therefore for inference we unpack the dataset into slices along each axis, crop,
        normalize, and store for inference along with the metadata needed to reconstruct
        the probability map in the problem space.
        ''' 
        img         = nib.load(case[0])
        case        = np.stack([nib.load(img).get_fdata() for img in case])
        brain_crop, nonzero_brain, orig_shape_brain  = self.crop_to_brain(case)
        sagittal_slices, coronal_slices, axial_slices = \
                self.slice_dataset(brain_crop)
        return {
                'nonzero_brain': nonzero_brain,
                'orig_shape_brain': orig_shape_brain,
                'header': img.header,
                'affine': img.affine,
                'sagittal_slices': [self.crop_to_brain_and_normalize(slc.squeeze(), num_downs)\
                        for slc in sagittal_slices],
                'coronal_slices': [self.crop_to_brain_and_normalize(slc.squeeze(), num_downs)\
                        for slc in coronal_slices],
                'axial_slices': [self.crop_to_brain_and_normalize(slc.squeeze(), num_downs)\
                        for slc in axial_slices]
                }
        
    def __len__(self):
        return len(self.filenames[0])

    def __getitem__(self, idx):
        case    = [mode[idx] for mode in self.filenames]
        return self.read_brain(case)

