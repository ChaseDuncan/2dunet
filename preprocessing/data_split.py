import os
import random
from shutil import copyfile
from glob import glob

''' split the brats 2020 dataset into random 80/20 split. '''
tr_dir ='brats2020/2dunet/train/'
te_dir ='brats2020/2dunet/test/'
os.makedirs(tr_dir, exist_ok=True)
os.makedirs(te_dir, exist_ok=True)

filenames       = glob(f'brats2020/MICCAI_BraTS2020_TrainingData/*/*.nii.gz')
filenames       = [[f for f in filenames if 't1.' in f],
                [f for f in filenames if 't1ce.' in f],
                [f for f in filenames if 't2.' in f],
                [f for f in filenames if 'flair.' in f],
                [f for f in filenames if 'seg.' in f]
                ]
z               = list(zip(*filenames))
random.shuffle(z)
sp = round(.8*len(filenames[0]))
train, test = z[:sp], z[sp:]
for f1, f2, f3, f4, f5 in train:
    copyfile(f1, os.path.join(tr_dir, f1.split('/')[-1]))
    copyfile(f2, os.path.join(tr_dir, f2.split('/')[-1]))
    copyfile(f3, os.path.join(tr_dir, f3.split('/')[-1]))
    copyfile(f4, os.path.join(tr_dir, f4.split('/')[-1]))
    copyfile(f5, os.path.join(tr_dir, f5.split('/')[-1]))

for f1, f2, f3, f4, f5 in test:
    copyfile(f1, os.path.join(te_dir, f1.split('/')[-1]))
    copyfile(f2, os.path.join(te_dir, f2.split('/')[-1]))
    copyfile(f3, os.path.join(te_dir, f3.split('/')[-1]))
    copyfile(f4, os.path.join(te_dir, f4.split('/')[-1]))
    copyfile(f5, os.path.join(te_dir, f5.split('/')[-1]))
