import os
import tables
import numpy as np
import nibabel as nib

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

from utils import *

def read_brain(case):
    ''' Expects 5xHxWxD tensor where patient[-1] is the segmentation.'''
    # since we throw out slices that have no labeled voxels, we cannot
    # know ahead of time what size the output will be.
    examples = []
    shapes = []

    # there's probably a cleaner way to do this... 
    for i in range(case.shape[1]):
        # check if slice has no labeled voxels
        if len(np.where(case[-1, i, :, :] > 0)[0]) == 0:
            continue
        else:# slice has labels. preprocess it. add it to the list.
            # crop to brain
            nonzero = np.where(case[:-1, i] != 0)
            x, y = nonzero[1], nonzero[2]
            # do this at the end?
            if x.shape[0] == 0: # not all cases have every mode
                continue

            brain_crop = case[:, i, np.min(x):np.max(x)+1, np.min(y):np.max(y)+1].copy()
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]))
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)

            examples.append(brain_crop)
            shapes.append(orig_shape)

    for i in range(case.shape[2]):
        # check if slice has no labeled voxels
        if len(np.where(case[-1, :, i] > 0)[0]) == 0:
            continue
        else:# slice has labels. preprocess it. add it to the list.
            # crop to brain
            nonzero = np.where(case[:-1, :, i] != 0)
            x, y = nonzero[1], nonzero[2]
            if x.shape[0] == 0: # not all cases have every mode
                continue
            # do this at the end?

            brain_crop = case[:, np.min(x):np.max(x)+1, i, np.min(y):np.max(y)+1].copy()
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]))
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)
            examples.append(brain_crop)
            shapes.append(orig_shape)

    for i in range(case.shape[3]):
        # check if slice has no labeled voxels
        if len(np.where(case[-1, :, :, i] > 0)[0]) == 0:
            continue
        else:# slice has labels. preprocess it. add it to the list.
            # crop to brain
            nonzero = np.where(case[:-1, :, :, i] != 0)
            x, y = nonzero[1], nonzero[2]
            if x.shape[0] == 0: # not all cases have every mode
                continue

            # do this at the end?
            brain_crop = case[:, np.min(x):np.max(x)+1, np.min(y):np.max(y)+1, i].copy()
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]) + 1e-32)
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)
            examples.append(brain_crop)
            shapes.append(orig_shape)

    return examples, shapes

def preprocess(input_dir, output_dir):
    filenames       = glob(input_dir)
    filenames       = [[f for f in filenames if 't1.' in f],
                    [f for f in filenames if 't1ce.' in f],
                    [f for f in filenames if 't2.' in f],
                    [f for f in filenames if 'flair.' in f],
                    [f for f in filenames if 'seg.' in f]
                    ]

    meta_storage    = []
    data_dir = os.path.join(output_dir, 'data/')
    os.makedirs(data_dir, exist_ok=True)

    with tqdm(total=len(filenames[0])) as pbar:
        for patient in zip(*filenames):
            pid = patient_id(patient[0])
            case = np.stack([nib.load(img).get_fdata() for img in patient])
            examples_p, shapes_p = read_brain(case)
            for i, e in enumerate(examples_p):
                # save example
                np.save(os.path.join(data_dir, f'{pid}.{i:03}.npy'), e)

            # TODO: metadata
            #for s in shapes_p:
            #    meta_storage.append(s[np.newaxis, ...])

            pbar.update(1)

if __name__=='__main__':
    input_dir = 'brats2020/MICCAI_BraTS2020_TrainingData/*/*.nii.gz'
    output_dir = 'brats2020/2d-preprocessed/'
    os.makedirs(output_dir, exist_ok=True)
    preprocess(input_dir, output_dir)

