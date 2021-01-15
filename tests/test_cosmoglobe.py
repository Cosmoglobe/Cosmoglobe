from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe
import h5py
import os
import unittest
import numpy as np 
import pathlib

dir_path = 'chain_test'
h5_file = 'chain_test.h5'
h5_file_path = os.path.join(dir_path, h5_file)

comps = ['ame', 'bandpass', 'cmb', 'dust', 'ff', 'gain', 'md',
         'radio', 'synch']

default_data_dir = pathlib.Path.cwd().parent.joinpath('cosmoglobe/data')
default_data_file = default_data_dir.joinpath('chain_c0001.h5')

class TestCosmoglobe(unittest.TestCase):

    def setUp(self):
        os.mkdir(dir_path)
        with h5py.File(h5_file_path,'w') as f:
            g = f.create_group('000001')
            for comp in comps:
                g.create_group(comp)

        self.sky_1 = Cosmoglobe()
        self.sky_2 = Cosmoglobe(dir_path)
        self.sky_3 = Cosmoglobe(h5_file_path)

    def tearDown(self):
        os.remove(h5_file_path)
        os.rmdir(dir_path)

    def test_datafile(self):
        self.assertEqual(self.sky_1.datafile, default_data_file)
        self.assertEqual(self.sky_2.datafile, pathlib.Path(h5_file_path))
        self.assertEqual(self.sky_3.datafile, pathlib.Path(h5_file_path))

    def test_components(self):   
        self.assertEqual(self.sky_1.components, comps)
        self.assertEqual(self.sky_2.components, comps)
        self.assertEqual(self.sky_3.components, comps)


if __name__ == "__main__":
    unittest.main()