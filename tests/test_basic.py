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

from cosmoglobe.plot import plot

# from cosmoglobe.sky import SkyModel

# model = SkyModel(nside=64)
# model = SkyModel(nside=2048, comps=['dust', 'ff', 'synch'], release='8')


hp.disable_warnings()

data_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/")
chain_dir = pathlib.Path("/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/")
chain = chain_dir / "chain_test.h5"
# chain = chain_dir / "bla.h5"
bandpass = chain_dir / "wmap_bandpass.txt"
wmap = hp.read_map(chain_dir / 'wmap_band_iqusmap_r9_9yr_K_v5.fits')

model = model_from_chain(chain, nside=256)
# model = model_from_chain(chain, nside=256, burn_in=20)
# model.remove('radio')
bp_freqs, bp, _ = np.loadtxt(bandpass, unpack=True)
bp_freqs*= u.GHz
bp *= u.K
# model(fwhm)

# model.dust(freq, fwhm)
# model.disable('cmb')
dipole = model.cmb.remove_dipole(return_dipole=True)
# hp.mollview(dipole, min=-3400, max=3400)
# hp.mollview(model.cmb.amp[0], min=-200, max=200)

# model.disable('radio')

# model.dust.spectral_parameters['T'] = np.random.randint(low=10 , high= 50,size=np.shape(model.synch.spectral_parameters['beta']))*u.K
# model.dust.spectral_parameters['beta'] = np.random.randint(low=-3 , high= -1,size=np.shape(model.synch.spectral_parameters['beta']))*u.dimensionless_unscaled
# print(model.dust.spectral_parameters['T'])
# print(model.dust.spectral_parameters['beta'])
# exit()
# freqs = np.flip(np.logspace(0.1, 2.7, 12)*u.GHz)
# print(model.dust.spectral_parameters)
# emissions = model(bp_freqs, bp, output_unit='MJy/sr')
# emissions = model(freqs, fwhm=30*u.arcmin, output_unit='MJy/sr')
# print(np.shape(emissions))
# print(emissions)
# hp.mollview(emissions[0], norm='hist')
# for freq, emission in zip(freqs, emissions):
    # hp.mollview(emission[2], norm='hist', title=f'{int(freq.value)}')
# hp.mollview(model(10*u.GHz, fwhm=20*u.arcmin)[0], norm='hist')
# hp.mollview(model.ame.amp[0], norm='hist')
# hp.mollview(model(353*u.GHz, fwhm=30*u.arcmin)[0], norm='hist', cmap='CMRmap')
# hp.mollview(model(353*u.GHz, fwhm=30*u.arcmin)[0], norm='hist')
# hp.mollview(model.radio.get_map(model.radio.amp, nside=64, fwhm=50*u.arcmin)[0], norm='hist', cmap='CMRmap')
# hp.mollview(model(30*u.GHz, fwhm=0.88*u.deg)[0], min=-200, max=5000,)
# hp.mollview(model(100*u.GHz, fwhm=30*u.arcmin)[0], norm='hist')


# hp.mollview(model(50*u.GHz, fwhm=0.88*u.deg, output_unit='mK')[0], norm='hist')
# hp.mollview(model(bp_freqs, bp, fwhm=0.88*u.deg, output_unit='mK')[0], norm='hist')

# mollplot(model, freq=50*u.GHz, fwhm=30*u.arcmin)

plot(model, comp='synch')
plt.show()