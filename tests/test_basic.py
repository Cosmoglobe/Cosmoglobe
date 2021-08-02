from context import cosmoglobe
from cosmoglobe.sky import model_from_chain
from cosmoglobe import sky_model

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from sys import exit
import pathlib

from cosmoglobe import plot


data_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/")
chain_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/")
chain = chain_dir / "chain_test.h5"
# chain = chain_dir / "bla.h5"
bandpass = chain_dir / "wmap_bandpass.txt"
wmap = hp.read_map(chain_dir / 'wmap_band_iqusmap_r9_9yr_K_v5.fits')
bp_freqs, bp, _ = np.loadtxt(bandpass, unpack=True)
bp_freqs*= u.GHz
bp *= u.K
print(bp_freqs.mean())
# model = model_from_chain(chain, nside=256)

model = sky_model(nside=256)

print(model)
model.cmb.remove_dipole()
# model._add_component_to_model(10)
# hp.mollview(model.radio(40*u.GHz, output_unit='MJy/sr')[0], norm='hist')
hp.mollview(model(22*u.GHz, fwhm=0.88*u.deg, output_unit='MJy/sr')[0], norm='hist')
# hp.mollview(model(bp_freqs, bp, fwhm=0.88*u.deg, output_unit='MJy/sr')[0], norm='hist')
# hp.mollview(model.radio(bp_freqs, bp, fwhm=0.88*u.deg, output_unit='MJy/sr')[0], norm='hist')
# print(model(50*u.GHz, fwhm=30*u.arcmin))
plt.show()

u.brightness_temperature

