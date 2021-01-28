import os
import tables
import numpy as np
import nibabel as nib

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
import pickle

from utils import *

orientations = ['axial', 'coronal', 'sagittal']
def preprocess(input_dir, output_dir, no_crop):
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

            #   hacks for catching unexpected things
            for c in case:
                if c.size == 0:
                    import pdb; pdb.set_trace()
            if case.shape[0] < 5:
                import pdb; pdb.set_trace()

            brain_crop, nonzero, orig_shape = read_brain(case, no_crop=no_crop)
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
    argparser.add_argument('--no_crop', action='store_true', help='do not crop brain.')
    args = argparser.parse_args()
    preprocess(args.input_dir, args.output_dir, args.no_crop)

