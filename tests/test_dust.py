from context import cosmoglobe
from cosmoglobe.models.dust import(
    ModifiedBlackbody, 
    blackbody_emission,
    blackbody_ratio
)
from cosmoglobe.cosmoglobe import Cosmoglobe
import unittest
import numpy as np 
import healpy as hp 
import astropy.units as u

nside = 64
npix = hp.nside2npix(nside)


class TestModifiedBlackbody(unittest.TestCase):

    def setUp(self):
        self.sky = Cosmoglobe()
        self.dust = self.sky.model('ModifiedBlackbody', nside)        

    def test__init__(self):
        self.dust_test1 = self.sky.model('ModifiedBlackbody', nside)

        with self.assertRaises(ValueError):
            self.dust_test2 = self.sky.model('not_a_model', nside)
        with self.assertRaises(ValueError):
            self.dust_test3 = self.sky.model('ModifiedBlackbody', 10)
        with self.assertRaises(ValueError):
            self.dust_test4 = self.sky.model('ModifiedBlackbody', 64.0)        
        with self.assertRaises(ValueError):
            self.dust_test5 = self.sky.model('ModifiedBlackbody', '64')        
        with self.assertRaises(ValueError):
            self.dust_test5 = self.sky.model('ModifiedBlackbody', -64)
        

    def test_maps(self):
        self.assertEqual(np.shape(self.dust.amp)[0], 3)
        self.assertEqual(len(self.dust.amp[0]), npix)
        self.assertEqual(len(self.dust.amp[1]), npix)
        self.assertEqual(len(self.dust.amp[2]), npix)        
        
        self.assertEqual(np.shape(self.dust.beta)[0], 3)
        self.assertEqual(len(self.dust.beta[0]), npix)
        self.assertEqual(len(self.dust.beta[1]), npix)
        self.assertEqual(len(self.dust.beta[2]), npix)        
        
        self.assertEqual(np.shape(self.dust.T)[0], 3)
        self.assertEqual(len(self.dust.T[0]), npix)
        self.assertEqual(len(self.dust.T[1]), npix)
        self.assertEqual(len(self.dust.T[2]), npix)

    def test_blackbody_emission(self):
        self.assertAlmostEqual(
            blackbody_emission(30*u.GHz, 300*u.K).to(
                u.J/(u.s* u.m**2 *u.sr*u.Hz), equivalencies=u.brightness_temperature(30*u.GHz)
            ).value*1e17,
            8.27547659048, 5
        )        
        self.assertAlmostEqual(
            blackbody_emission(40*u.THz, 500*u.K).to(
                u.J/(u.s* u.m**2 *u.sr*u.Hz), equivalencies=u.brightness_temperature(40*u.THz)
            ).value*1e11,
            2.07414320812, 5
        )
        self.assertIsInstance(blackbody_emission(30*u.GHz, 300*u.K), u.quantity.Quantity)
        
        with self.assertRaises(TypeError):
            blackbody_emission(30, 300*u.K)        
        with self.assertRaises(TypeError):
            blackbody_emission(30*u.GHz, 300)
        with self.assertRaises(u.UnitsError):
            blackbody_emission(30*u.K, 300)
        with self.assertRaises(u.UnitsError):
            blackbody_emission(30*u.GHz, 300*u.m)

    def test_blackbody_ratio(self):
        self.assertIsInstance(blackbody_ratio(300*u.GHz, 400*u.GHz, 300*u.K), u.quantity.Quantity)
        self.assertTrue(0 <= 1/blackbody_ratio(300*u.GHz, 400*u.GHz, 300*u.K).value <= 1)
        self.assertTrue(0 <= 1/blackbody_ratio(100*u.MHz, 600*u.GHz, 300*u.K).value <= 1)
        self.assertTrue(0 <= 1/blackbody_ratio(30*u.MHz, 600*u.GHz, 300*u.mK).value <= 1)
        self.assertEqual(blackbody_ratio(30*u.GHz, 30*u.GHz, 300*u.mK).value, 1)

        with self.assertRaises(TypeError):
            blackbody_ratio(300, 400*u.GHz, 300*u.K)
        with self.assertRaises(TypeError):
            blackbody_ratio(300*u.GHz, 400, 300*u.K)
        with self.assertRaises(TypeError):
            blackbody_ratio(300*u.GHz, 400*u.GHz, 300)
        with self.assertRaises(TypeError):
            blackbody_ratio(300, 400, 300)
        with self.assertRaises(u.UnitsError):
            blackbody_ratio(30*u.m, 400*u.GHz, 300*u.K)
        with self.assertRaises(u.UnitsError):
            blackbody_ratio(30*u.GHz, 400*u.s, 300*u.K)
        with self.assertRaises(u.UnitsError):
            blackbody_ratio(30*u.m, 400*u.GHz, 300*u.J)

    def test_compute_emission(self):
        self.assertIsInstance(self.dust[30*u.GHz], u.quantity.Quantity)
        self.assertIsInstance(self.dust[30*u.MHz], u.quantity.Quantity)
        self.assertEqual(np.shape(self.dust[30*u.GHz])[0], 3)
        self.assertEqual(len(self.dust[30*u.GHz][0]), npix)
        self.assertEqual(len(self.dust[30*u.GHz][1]), npix)
        self.assertEqual(len(self.dust[30*u.GHz][2]), npix)
        self.assertEqual(self.dust[545*u.GHz][0].value.tolist(), self.dust.amp[0].value.tolist())

        with self.assertRaises(TypeError):
            self.dust[545]
        with self.assertRaises(u.UnitsError):
            self.dust[30*u.mm]



if __name__ == "__main__":
    unittest.main()