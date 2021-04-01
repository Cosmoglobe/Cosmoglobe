"""
For testing Cosmoglobe during development.

"""
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from context import cosmoglobe
import sys
hp.disable_warnings()
data = '../../Cosmoglobe_test_data/bla.h5'


import healpy as hp
from cosmoglobe.sky.components import PowerLaw, ModifiedBlackBody
from cosmoglobe.sky.model import Model
from cosmoglobe.tools.map import to_stokes, StokesMap
import cosmoglobe.tools.h5 as h5
import sys

map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'

bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
nu_array, bandpass_array, _ = np.loadtxt(bandpass_data, unpack=True)

nside = 512
# iqu = np.random.randint(low=1, high=5, size=(3, hp.nside2npix(nside)))
intensity = np.array([np.ones(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside))])
pol = np.ones((3, hp.nside2npix(nside)))

# i = np.random.randint(low=1, high=5, size=(hp.nside2npix(nside)))
# i2 = np.ones(hp.nside2npix(nside))
i = np.ones(hp.nside2npix(nside))
scalar = 5


# synch = PowerLaw(comp_name='synch', amp=i*u.K, freq_ref=30*u.GHz, beta=i)
# dust = ModifiedBlackBody(comp_name='dust', amp=pol*u.K, freq_ref=[20,10,10]*u.GHz, beta=pol, T=i*u.K)



# model = Model([synch, dust], nside=nside)

model = h5.model_from_chain(file=data, sample=30)
print(model)

hp.mollview(model.get_emission(40*u.GHz).P, norm='hist')

plt.show()