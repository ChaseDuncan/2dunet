import os
import tables
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

def read_brain(case):
    ''' Expects 5xHxWxD tensor where patient[-1] is the segmentation.'''
    # since we throw out slices that have no labeled voxels, we cannot
    # know ahead of time what size the output will be.
    examples = []
    shapes = []
    pad_size = (7, 240, 240)
    def binarize_problem(multilabel_problem):
        et = np.zeros(multilabel_problem.shape)
        wt = np.zeros(multilabel_problem.shape)
        tc = np.zeros(multilabel_problem.shape)
        et[np.where(multilabel_problem == 4)] = 1
        wt[np.where(multilabel_problem > 0)] = 1
        tc[np.where((multilabel_problem == 4) | (multilabel_problem == 1))] = 1
        return np.stack([et, wt, tc])

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

            brain_crop = case[:, i, np.min(x):np.max(x)+1, np.min(y):np.max(y)+1]
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]))
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)
            ex = np.zeros(pad_size)
            ex[:, :brain_crop.shape[1], :brain_crop.shape[2]] = brain_crop
            examples.append(ex)
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
            brain_crop = case[:, np.min(x):np.max(x)+1, i, np.min(y):np.max(y)+1]
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]))
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)
            ex = np.zeros(pad_size)
            ex[:, :brain_crop.shape[1], :brain_crop.shape[2]] = brain_crop
            examples.append(ex)
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
            brain_crop = case[:, np.min(x):np.max(x)+1, np.min(y):np.max(y)+1, i]
            brain_crop[:-1] = (brain_crop[:-1] - np.min(brain_crop[:-1])) / (np.max(brain_crop[:-1]) - np.min(brain_crop[:-1]) + 1e-32)
            brain_crop = np.concatenate([brain_crop[:-1], binarize_problem(brain_crop[-1])])
            orig_shape = np.array(brain_crop[0].shape)
            ex = np.zeros(pad_size)
            ex[:, :brain_crop.shape[1], :brain_crop.shape[2]] = brain_crop
            examples.append(ex)
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
    hdf5_file       = tables.open_file(os.path.join(output_dir + 'test.data.hdf5'), mode='w')

    modes           = 7
    side            = 240

    # dummy dimension for exlarging over
    data_shape      = (0, modes, side, side)
    meta_shape      = (0, 2)

    # earray is 'enlargeable' meaning new elements can be added to it on disk
    # in any dimension but only one dimension.
    data_storage    = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                            expectedrows=1)
                            #expectedrows=(300*(240**2)*155))
    meta_storage    = hdf5_file.create_earray(hdf5_file.root, 'meta', tables.UInt8Atom(), shape=meta_shape,
                            expectedrows=1)
                            #expectedrows=(300*(240**2)*155))
    #patient = [f[90] for f in filenames] 
    #case = np.stack([nib.load(img).get_fdata() for img in patient])
    #examples_p, shapes_p = read_brain(case)
    #for e in examples_p:
    #    data_storage.append(e[np.newaxis, ...]) 
    #for s in shapes_p:
    #    meta_storage.append(s[np.newaxis, ...])

    with tqdm(total=len(filenames[0])) as pbar:
        for patient in zip(*filenames):
            case = np.stack([nib.load(img).get_fdata() for img in patient])
            examples_p, shapes_p = read_brain(case)
            for e in examples_p:
                data_storage.append(e[np.newaxis, ...]) 
            for s in shapes_p:
                meta_storage.append(s[np.newaxis, ...])
            pbar.update(1)
    hdf5_file.close() 

if __name__=='__main__':
    # glob
    input_dir = 'brats2020/MICCAI_BraTS2020_TrainingData/*/*.nii.gz'
    output_dir = 'brats2020/2dhdf5/'
    os.makedirs(output_dir, exist_ok=True)
    preprocess(input_dir, output_dir)

