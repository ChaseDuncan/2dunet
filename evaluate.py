from argparse import ArgumentParser
from glob import glob
import pickle
import sys 
import os
import numpy as np
from tqdm import tqdm
from utils import *
import nibabel as nib

pred_dir = 'brats2020/2dunet/model-predictions/deleteme/'
in_dir = 'brats2020/2dunet/model-predictions/deleteme/volumes/'
sg_dir = 'brats2020/2dunet/test/'

patients = sorted(glob(os.path.join(in_dir, '*')))
def nll(pred, seg):
    nll_ = np.dot(
            -np.log(pred).reshape(np.prod(pred.shape)),
            seg.reshape(np.prod(seg.shape))
            )
    z = np.dot(seg.reshape(np.prod(seg.shape)), seg.reshape(np.prod(seg.shape)))
    return nll_ / (z + 1e-16)

def soft_dice(pred, seg):
    den_of_evil = np.dot(seg.reshape(np.prod(seg.shape)), seg.reshape(np.prod(seg.shape)))\
            + np.dot(pred.reshape(np.prod(seg.shape)), pred.reshape(np.prod(seg.shape)))+1e-16
    return 2*np.dot(seg.reshape(np.prod(seg.shape)), pred.reshape(np.prod(seg.shape)))/den_of_evil

def dice(sigs, segs, threshold=0.5):
    #   transform prediction into binary then call soft_dice
    preds = np.zeros(sigs.shape)
    preds[np.where(sigs > threshold)] = 1
    return soft_dice(preds, segs)

micro_dice = []
micro_nll  = []
for i, patient in enumerate(patients):
    seg_nifti_path = os.path.join(sg_dir, f'{get_pid(patient)}_seg.nii.gz')
    pred_npy_path = os.path.join(patient, '3d_all_orientations.npy')
    preds = np.load(pred_npy_path)
    #   extend orientations list using combinations of  
    orientations = [pred.squeeze() for pred in preds]
    orientations.append((orientations[0] + orientations[2]) / 2)
    orientations.append((orientations[0] +orientations[1] + orientations[2]) / 3)

    segs = nib.load(seg_nifti_path).get_fdata()
    segs = binarize_problem(segs)
    micro_dice.append([[dice(tumor_type, seg)\
            for tumor_type, seg in zip(orientation, segs)] \
            for orientation in orientations ])
    micro_nll.append([[nll(tumor_type, seg)\
            for tumor_type, seg in zip(orientation, segs)] \
            for orientation in orientations ])

mda = np.array(micro_dice)
mna = np.array(micro_nll)
avg_dice = [np.sum(mda[:, i], axis=0)/mda.shape[0] for i in range(mda.shape[1])]
avg_nll = [np.sum(mna[:, i], axis=0)/mna.shape[0] for i in range(mna.shape[1])]
print(avg_dice)
print(avg_nll)
