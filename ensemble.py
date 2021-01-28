from argparse import ArgumentParser
from glob import glob
import pickle
import sys 
import os
import numpy as np
from tqdm import tqdm
from utils import get_pid

pred_dir = 'brats2020/2dunet/model-predictions/deleteme/'
in_dir = 'brats2020/2dunet/model-predictions/deleteme/slices/'
ot_dir = 'brats2020/2dunet/model-predictions/deleteme/volumes/'
patients = glob(os.path.join(in_dir, '*'))
#meta_data = glob('brats2020/2dunet/test-preprocessed-v2/meta/*')

def read_metadata(patient):
    fmeta_data = [ meta for meta in meta_data if patient in meta][0]
    pmeta_data = pickle.load(open(fmeta_data, 'rb'))
    return pmeta_data['header']


with tqdm(total=len(patients)) as pbar:
    for patient in patients:
        pid = get_pid(patient)
        seg_files = glob(os.path.join(patient, '*'))
        orientations = ['axial', 'coronal', 'sagittal']
        swaps = [[0,1], [0,0], [2,3]]

        preds = []
        for i, orientation in enumerate(orientations):
            ax=i+1
            sig = np.concatenate(
                    [np.expand_dims(np.load(slc), axis=ax) for slc in seg_files if orientation in slc], 
                    axis=ax)
            sig = np.swapaxes(sig.squeeze(), swaps[i][0], swaps[i][1])
            preds.append(sig[None,])

        p_path = os.path.join(ot_dir, pid)
        os.makedirs(p_path, exist_ok=True)
        np.save(os.path.join(p_path, f'3d_all_orientations'), np.concatenate(preds))
        pbar.update(1)
