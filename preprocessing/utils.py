import numpy as np

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

