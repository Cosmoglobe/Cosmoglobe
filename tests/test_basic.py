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
import cosmoglobe.sky as sky
from cosmoglobe.tools.map import to_stokes
from cosmoglobe.tools.h5 import model_from_chain

map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'

bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
nu_array, bandpass_array, _ = np.loadtxt(bandpass_data, unpack=True)

nside = 64
iqu = np.random.randint(low=1, high=5, size=(3, hp.nside2npix(nside)))
# iqu = np.ones((3, hp.nside2npix(nside)))
i = np.random.randint(low=1, high=5, size=(hp.nside2npix(nside)))
# i = np.ones(hp.nside2npix(nside))
scalar = 5


# dust = sky.BlackBodyCMB('dust', amp=i*u.uK)
# print(dust.amp)
# model = model_from_chain(data, nside=16, burn_in=20)
# print(model)
# emission = model.get_emission([10, 500]*u.GHz)
# hp.mollview(emission[0].I, norm='hist')
# hp.mollview(emission[1].I, norm='hist')

i_30 = hp.smoothing(i, (30*u.arcmin).to(u.rad).value, use_pixel_weights=True)
i_60 = hp.smoothing(i_30, (np.sqrt(60**2 - 30**2)*u.arcmin).to(u.rad).value, use_pixel_weights=True)
i__60 = hp.smoothing(i, (60*u.arcmin).to(u.rad).value, use_pixel_weights=True)

print(i_60 - i__60)
hp.mollview(i_60)
hp.mollview(i__60)
hp.mollview(i_60-i__60)
plt.show()