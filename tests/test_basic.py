"""
For testing Cosmoglobe during development.

"""
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from context import cosmoglobe

hp.disable_warnings()
data = '../../Cosmoglobe_test_data/bla.h5'


import healpy as hp
from cosmoglobe.sky.components import PowerLaw, ModifiedBlackBody
from cosmoglobe.sky.model import Model
from cosmoglobe.tools.map import IQUMap
map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'

bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
nu_array, bandpass_array, _ = np.loadtxt(bandpass_data, unpack=True)

nside = 32
iqu = np.random.randint(low=1, high=5, size=(3, hp.nside2npix(nside)))
# iqu = np.ones((3, hp.nside2npix(nside)))
# i = np.random.randint(low=1, high=5, size=(hp.nside2npix(nside)))
i = np.ones(hp.nside2npix(nside))
scalar = 5

synch = PowerLaw(amp=iqu, nu_ref=[30, 30, 30]*u.GHz, beta=iqu)
dust = ModifiedBlackBody(amp=iqu*u.uK, nu_ref=[20,20,20]*u.GHz, beta=iqu, T=i*u.K)
model = Model(components=[('synch', synch), ('dust', dust)])

model.dust.get_emission(nu_array*u.GHz, bandpass_array*u.K)

map_ = IQUMap(I=i, Q=i, U=i, unit=u.K, nu_ref=3*u.GHz, label="Test map")
# print(_get_interp_range(dust.spectrals['beta'], 10))

