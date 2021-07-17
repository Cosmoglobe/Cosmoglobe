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

# model = model_from_chain(chain, nside=256)

# plt.plot

model = sky_model(nside=256)
# print(help(model))
# emission = model.dust(50*u.GHz, fwhm=50*u.arcmin, output_unit='MJy/sr')
# print(emission[0])
# np.linspace
model.cmb.remove_dipole()
# model.disable('radio')
hp.mollview(model(bp_freqs, bp, fwhm=0.88*u.deg, output_unit='uK_RJ')[0], unit='uK_RJ', norm='hist')
# hp.mollview(model.cmb(30*u.GHz, fwhm=0.88*u.deg, output_unit='uK_CMB')[0], norm='hist')
# hp.mollview(model.cmb(100*u.GHz, fwhm=0.88*u.deg, output_unit='uK_CMB')[0], norm='hist')
# hp.mollview(model.cmb(400*u.GHz, fwhm=0.88*u.deg, output_unit='uK_CMB')[0], norm='hist')
# hp.mollview(model(bp_freqs, bp, fwhm=60*u.arcmin, output_unit='MJy/sr')[0], unit='MJy/sr', norm='hist')
# hp.mollview(model(bp_freqs, bp, fwhm=60*u.arcmin, output_unit='uK_CMB')[0], unit='uK_CMB', norm='hist')
# hp.mollview(model(bp_freqs, bp, fwhm=60*u.arcmin)[0],unit='uK_RJ', norm='hist')
# hp.mollview(emission[0], norm='hist')

# plot(model.dust(100.5*u.GHz,))
# plt.show()
# freqs = u.Quantity(np.arange(1, 3), unit=u.GHz)
# print(model.synch.get_emission(freqs, fwhm=10*u.arcmin, output_unit='uK_CMB'))
# chain_to_h5(chainfile=chain, output_dir='/Users/metinsan/Documents/doktor/models/test1')
# model = model_from_h5('/Users/metinsan/Documents/doktor/models/test1/model_512.h5')
# print(model)
# model_to_h5(model, dirname)
# model_from_h5(filename)
# chain_to_h5(chain, dirname)

# plot(model, comp="dust")
# plot(model, comp="ff")
# plot(model, comp="synch")
# plot(model, comp="ame")
# plot(model, comp="cmb")
# plot(model, freq=100*u.GHz, fwhm=20*u.arcmin)
# emission = model(bp_freqs, bp, fwhm=30*u.arcmin, output_unit='MJy/sr')[0]
# plot(emission)
# plt.show()
