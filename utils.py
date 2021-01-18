import numpy as np
import torch
import torch.nn as nn
from time import time

def dim_diff(dim, num_downs):
    mult    = 2**num_downs*np.ceil(dim / 2**num_downs)
    return int(mult - dim)

def pad_ex(ex, mult):
    pad = ((0, 0), (0, int(mult - ex.shape[-2])), (0, int(mult - ex.shape[-1])))
    ex_pad = np.pad(ex, pad)
    return ex_pad

num_downs = 3
def collate_fn(batch):
    ''' all examples in a batch must have the same size. all sides must be divisible by 8.'''
    max_dim_size   = max([max(x.size()) for x in batch])
    mult    = max(int(2**num_downs*np.ceil(max_dim_size / 2**num_downs)), 128)
    pad_l=[pad_ex(ex, mult).unsqueeze(0) for ex in batch] 
    padded_batch = torch.cat(pad_l)
    return padded_batch[:, :4], padded_batch[:, 4:]

def binarize_problem(multilabel_problem):
    et = np.zeros(multilabel_problem.shape)
    wt = np.zeros(multilabel_problem.shape)
    tc = np.zeros(multilabel_problem.shape)
    et[np.where(multilabel_problem == 4)] = 1
    wt[np.where(multilabel_problem > 0)] = 1
    tc[np.where((multilabel_problem == 4) | (multilabel_problem == 1))] = 1
    return np.stack([et, wt, tc])

def patient_id(img_file):
    ''' gets the patient id from a brats2020 file. here because it's ugly. '''
    return "_".join(img_file.split("/")[-1].split("_")[:-1])

def normalize(case): 
    ''' important: assumes ground truth is -1th volume '''
    norm_case = np.zeros(case.shape)
    norm_case[-1] = case[-1]
    for i, mode in enumerate(case[:-1]):
        norm_case[i] = ( mode - np.min(mode) ) / ( np.max(mode) - np.min(mode) )
    return norm_case

def nonzero_coords(imgs_npy): 
    ''' gives the nonzero coordinates of a particular mode'''
    nonzero = []
    for i in imgs_npy[:4]:
        w = np.array(np.where(i != 0)) 
        if w.size != 0:
            nonzero.append(w)
    if len(nonzero) == 0:
        return nonzero
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    return nonzero

def crop_to_brain(case, img_size=200):
    ''' note: this relies on case having 4 dimensions. unsqueeze as necessary.'''
    orig_shape  = case.shape
    nonzero     = nonzero_coords(case)
    # img is all zeros
    if len(nonzero) == 0:
        crop = case.squeeze()
    else:
        crop = case[:, nonzero[0][0] : nonzero[0][1] + 1,
                       nonzero[1][0] : nonzero[1][1] + 1,
                       nonzero[2][0] : nonzero[2][1] + 1].squeeze()
    crop = pad_ex(crop, img_size)
    return crop, nonzero, orig_shape

def crop_to_brain_and_normalize(case): 
    brain_crop, nonzero, orig_shape = crop_to_brain(case)
    return normalize(brain_crop), nonzero, orig_shape


def read_brain(case):
    brain_crop, nonzero, orig_shape = crop_to_brain_and_normalize(case)
    brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
    return brain_crop, nonzero, orig_shape

def has_tumor(seg):
    if np.sum(seg):
        return 'tum'
    else:
        return 'not'
    
def slice_dataset(case):
    # i don't know if this copy is actually needed here. 
    slices = [ np.split(case.copy(), case.shape[i+1], axis=i+1) for i in range(len(case.shape)-1) ]
    return slices


