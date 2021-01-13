from dataloader import BraTS2020Test2d
import numpy as np

class test_BraTS2020Test2d(BraTS2020Test2d):
    def __init__(self, test_data_dir):
        super().__init__(test_data_dir)
        
    def nonzero_coords_t(self):
        ''' Make sure nonzero_coords returns the numbers we expect.'''
        X           = np.arange(60).reshape((3,4,5))
        X           = np.pad(X, ((7, 33), (9999, 3), (1, 0)))
        expected    = [[7, 10], [9999, 10003], [1, 6]]
        # test for size check
        #X = np.arange(81).reshape((3,3,3,3))
        nonzero     = self.nonzero_coords(X)
        assert  nonzero == expected, f'expected: {expected} \t got: {nonzero}'

    def crop_to_brain_t(self):
        d           = np.arange(4*9).reshape((4,3,3))
        d_padded    = np.pad(d, ((0, 0), (3248957, 34), (0, 234)))
        d_crop, _      = self.crop_to_brain(d_padded)
        assert np.array_equal(d, d_crop), f'expected: {d} \t got: {d_crop}'


if __name__=='__main__':
    test_dataset = test_BraTS2020Test2d('brats2020/MICCAI_BraTS2020_ValidationData/*/*.nii.gz')
    test_dataset.read_brain([f[0] for f in test_dataset.filenames])
    test_dataset.nonzero_coords_t()
    test_dataset.crop_to_brain_t()

    
