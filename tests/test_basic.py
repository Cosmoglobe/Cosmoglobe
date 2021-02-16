"""
For testing Cosmoglobe during development.

"""
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from context import cosmoglobe
from cosmoglobe.skymodel import SkyModel

hp.disable_warnings()
planck_cmap = ListedColormap(np.loadtxt('../../Cosmoglobe_test_data/planck_cmap.dat')/255.)
data = '../../Cosmoglobe_test_data/bla.h5'
# data = '../../Cosmoglobe_test_data/data.json'
bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'

nus, bandpass1, _ = np.loadtxt(bandpass_data, unpack=True)


sky = SkyModel(data, fwhm=150*u.arcmin, components=['cmb'])

hp.mollview(sky.cmb.amp[0], min=-300, max=300, cmap=planck_cmap)
plt.show()



# freqs, rms = sky.get_spectrum(sky_frac=50)
# for model, model_rms in rms.items():
#     plt.loglog(freqs, model_rms, label=model)

# plt.ylim(5e-2, 1e3)
# plt.xlabel('Frequency [GHz]')
# plt.ylabel('RMS brightness temperature [uK_RJ]')
# plt.legend()
# plt.show()

