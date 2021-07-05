from context import cosmoglobe
from cosmoglobe.sky import model_from_chain
from cosmoglobe.hub import save_model, load_model
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from sys import exit
import sys
import pathlib


# from cosmoglobe.sky import SkyModel

# model = SkyModel(nside=64)
# model = SkyModel(nside=2048, comps=['dust', 'ff', 'synch'], release='8')


hp.disable_warnings()

data_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/")
chain_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/")
chain = chain_dir / "chain_test.h5"
# chain = chain_dir / "bla.h5"
bandpass = chain_dir / "wmap_bandpass.txt"

model = model_from_chain(chain, nside=256)
# model = model_from_chain(chain, nside=256, burn_in=20)
# model.remove('radio')
bp_freqs, bp, _ = np.loadtxt(bandpass, unpack=True)
bp_freqs*= u.GHz
bp *= u.K
# model(fwhm)
# model.dust(freq, fwhm)
model.disable('cmb')
# model.disable('radio')

# freqs = np.flip(np.logspace(0.1, 2.7, 12)*u.GHz)
# print(model.dust.spectral_parameters)
# emissions = model(bp_freqs, bp, output_unit='MJy/sr')
# emissions = model(freqs, fwhm=20*u.arcmin, output_unit='MJy/sr')
# print(np.shape(emissions))
# print(emissions)
# hp.mollview(emissions[0], norm='hist')
# for freq, emission in zip(freqs, emissions):
#     hp.mollview(emission[0], norm='hist', title=f'{int(freq.value)}')
# hp.mollview(model(10*u.GHz, fwhm=20*u.arcmin)[0], norm='hist')

# hp.mollview(model(31*u.GHz, fwhm=30*u.arcmin)[0], norm='hist', cmap='CMRmap')
# hp.mollview(model.radio.get_map(model.radio.amp, nside=64, fwhm=50*u.arcmin)[0], norm='hist', cmap='CMRmap')
# hp.mollview(model(30*u.GHz, fwhm=0.88*u.deg)[0], min=-200, max=5000,)
# hp.mollview(model(100*u.GHz, fwhm=30*u.arcmin)[0], norm='hist')
hp.mollview(model(bp_freqs, bp, fwhm=0.88*u.deg, output_unit='MJy/sr')[0],norm='hist')

# model(50*u.GHz, )
# nside = 64
# lon, lat = np.loadtxt(point_src, usecols=(0,1), unpack=True)

# map_ = np.zeros(hp.nside2npix(nside))
# pixels = hp.ang2pix(nside, lon, lat, lonlat=True)
# print(np.shape(pixels))
# map_[pixels] += 1
# map_ = hp.smoothing(map_, (30*u.arcmin.to(u.rad)))

# hp.mollview(map_)
# # hp.projscatter(lon, lat, lonlat=True)

plt.show()

# cg.mollview(model.dust)