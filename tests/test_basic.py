"""For testing Cosmoglobe during development."""
from context import cosmoglobe
import cosmoglobe.sky as sky
import astropy.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from cosmoglobe.science.functions import K_CMB_to_K_RJ
import matplotlib.colors as col
import matplotlib as mpl

cmap = col.ListedColormap(np.loadtxt('planck_cmap.txt') / 255.0, "planck")
mpl.cm.register_cmap(name='planck', cmap=cmap)
hp.disable_warnings()

data = '../../Cosmoglobe_test_data/bla.h5'
map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'
bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
nu_array, bandpass_array, _ = np.loadtxt(bandpass_data, unpack=True)
nu_array *= u.GHz
bandpass_array *= u.uK

nside = 64
model = sky.model_from_chain(data, nside=nside, sample=None)

# for comp in model:
#     for idx, col in enumerate(comp.amp):
#         hp.mollview(col, norm='hist', title=f"{comp.name} {idx}", unit=col.unit, cmap='planck')

freq = 44*u.GHz
emission = model.get_emission([30,55]*u.GHz, output_unit='uK_CMB')
hp.mollview(emission[0][0], min=-3400, max=3400, title=f"full sky at {freq}", unit=emission[0].unit, cmap='planck')
# hp.mollview(emission[1], min=-34, max=34, title=f"full sky at {freq}", unit=emission[0].unit, cmap='planck')
# hp.mollview(emission[2], min=-34, max=34, title=f"full sky at {freq}", unit=emission[0].unit, cmap='planck')

plt.show()