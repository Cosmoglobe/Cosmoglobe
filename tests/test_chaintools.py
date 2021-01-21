from context import cosmoglobe
import cosmoglobe.chaintools as ctools
import pathlib
import unittest
import numpy as np
import healpy as hp

data_dir = pathlib.Path.cwd().parent.joinpath('cosmoglobe/data')
data_file = data_dir.joinpath('chain_c0001.h5')
nside = 64
npix = hp.nside2npix(nside)

class TestChaintools(unittest.TestCase):

    def test_get_chainfile(self):
        self.assertEqual(ctools.get_chainfile(data_dir), data_file)
        self.assertEqual(ctools.get_chainfile(data_file), data_file)

        with self.assertRaises(FileNotFoundError):
            ctools.get_chainfile('test_file.fits')
            ctools.get_chainfile('test_data')
        with self.assertRaises(TypeError):
            ctools.get_chainfile(5.0)

    def test_get_component_list(self):
        component_list = ['ame', 'bandpass', 'cmb', 'dust', 'ff',
                          'gain', 'md', 'radio', 'synch',]

        self.assertListEqual(
            ctools.get_component_list(data_file), component_list
        )

    def test_get_alm_list(self):
        dust_list = ['T', 'amp', 'beta',]
        synch_list = ['amp', 'beta',]
        ff_list = ['amp',]
        cmb_list = ['amp',]

        self.assertEqual(ctools.get_alm_list(data_file, 'dust'),
                         dust_list)        
        self.assertEqual(ctools.get_alm_list(data_file, 'synch'),
                         synch_list)        
        self.assertEqual(ctools.get_alm_list(data_file, 'ff'),
                         ff_list)        
        self.assertEqual(ctools.get_alm_list(data_file, 'cmb'),
                         cmb_list)
        
        with self.assertRaises(ValueError):
            ctools.get_alm_list(data_file, 'bandpass')
        with self.assertRaises(ValueError):
            ctools.get_alm_list(data_file, 'gain')
        with self.assertRaises(ValueError):    
            ctools.get_alm_list(data_file, 'md')
        with self.assertRaises(ValueError):
            ctools.get_alm_list(data_file, 'radio')
        with self.assertRaises(KeyError):
            ctools.get_alm_list(data_file, 'not_a_valid_component')


    def test_get_alms(self):
        self.assertEqual(np.shape(ctools.get_alms(
            data_file, component='dust', nside=nside))[0], 3 
            )      
        self.assertEqual(np.shape(ctools.get_alms(
            data_file, component='ff', nside=nside))[0], 1 
            )
        self.assertEqual(len(ctools.get_alms(
            data_file, component='dust', nside=nside)[1]), npix 
            )

        with self.assertRaises(KeyError):
            ctools.get_alms(data_file, component='not_a_valid_component', 
                            nside=nside)
        with self.assertRaises(KeyError):
            ctools.get_alms(data_file, component='dust',
                            param='not_a_valid_param', nside=nside)


    def test_nside_isvalid(self):
        with self.assertRaises(ValueError):
            ctools.nside_isvalid(511)
        with self.assertRaises(ValueError):
            ctools.nside_isvalid(64.0)        
        with self.assertRaises(ValueError):
            ctools.nside_isvalid('64')
        with self.assertRaises(ValueError):
            ctools.nside_isvalid(-64)

if __name__ == "__main__":
    unittest.main()