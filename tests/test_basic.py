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

hp.disable_warnings()

data_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/")
chain_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/")
chain = chain_dir / "bla.h5"
bandpass = chain_dir / "wmap_bandpass.txt"
point_src = data_dir / "COM_AT20G_GB6_NVSS_PCCS2_nothreshold_v8.dat"

# model = model_from_chain(chain, nside=64)
model = model_from_chain(chain, nside=512, burn_in=20)

bp_freqs, bp, _ = np.loadtxt(bandpass, unpack=True)
bp_freqs*= u.GHz
bp *= u.K

# freqs = np.flip(np.logspace(0.1, 2.7, 12)*u.GHz)
# print(model.dust.spectral_parameters)
emissions = model(bp_freqs, bp, fwhm=20*u.arcmin, output_unit='MJy/sr')
# emissions = model(bp_freqs, bp, output_unit='MJy/sr')
# print(np.shape(emissions))
# print(emissions)
hp.mollview(emissions[0], norm='hist')
# for emission, freq in zip(emissions, freqs):
    # hp.mollview(emission[0], title=f'{int(freq.value)} GHz', norm='hist')


model(50*u.GHz, )
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