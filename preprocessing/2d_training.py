import os
import tables
import numpy as np
import nibabel as nib

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
import pickle


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
    nonzero = [np.array(np.where(i != 0)) for i in imgs_npy]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    return nonzero

def crop_to_brain(case):
    orig_shape  = case.shape
    nonzero     = nonzero_coords(case)
    return case[:, nonzero[0][0] : nonzero[0][1] + 1,
                   nonzero[1][0] : nonzero[1][1] + 1,
                   nonzero[2][0] : nonzero[2][1] + 1], nonzero, orig_shape

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

orientations = ['axial', 'coronal', 'sagittal']
def preprocess(input_dir, output_dir):
    ''' creates output_dir if it doesn't exist already '''
    filenames       = glob(input_dir)
    filenames       = [[f for f in filenames if 't1.' in f],
                    [f for f in filenames if 't1ce.' in f],
                    [f for f in filenames if 't2.' in f],
                    [f for f in filenames if 'flair.' in f],
                    [f for f in filenames if 'seg.' in f]
                    ]

    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meta'), exist_ok=True)

    with tqdm(total=len(filenames[0])) as pbar:
        for patient in zip(*filenames):
            case = [nib.load(img) for img in patient]
            header = case[0].header
            case = np.stack([c.get_fdata() for c in case])

            brain_crop, nonzero, orig_shape = read_brain(case)
            brain_crop_slices = slice_dataset(brain_crop)

            pid = patient_id(patient[0])
            for orientation, slices in zip(orientations, brain_crop_slices):
                for i, slice in enumerate(slices):
                    filename = f'{pid}.{orientation}.{i:03}.{has_tumor(slice[-3:])}'
                    np.save(os.path.join(os.path.join(output_dir, 'data'), filename), slice)
                    pickle.dump(
                            { 'header': header, 
                            'nonzero': nonzero, 
                            'orig_shape': orig_shape},
                            open(os.path.join(os.path.join(output_dir, 'meta'), f'{filename}.pkl'), 'wb'))
            pbar.update(1)

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input_dir', type=str, help='directory where BraTS data is located.')
    argparser.add_argument('--output_dir', type=str, help='directory to store preprocessed data.')
    args = argparser.parse_args()
    preprocess(args.input_dir, args.output_dir)

