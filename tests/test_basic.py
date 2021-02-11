"""
For testing Cosmoglobe during development.

"""

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

hp.disable_warnings()

data = '../../Cosmoglobe_test_data/chain_test.h5'
bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'

nus, bandpass1, bandpass2 = np.loadtxt(bandpass_data, unpack=True)
# plt.plot(nus, bandpass1)
# plt.show()
sky = Cosmoglobe(data, sample='mean')
# cmb = sky.model('cmb')
cmb = sky.model('cmb', remove_dipole=True, remove_monopole=True)
synch = sky.model('synch')
dust = sky.model('dust')
ff = sky.model('ff')
ame = sky.model('ame')

# emission = cmb.get_emission(nus*u.GHz, bandpass1*u.K)
# hp.mollview(emission[0], norm='hist', title='bandpass')
# emission = cmb.get_emission(22*u.GHz)
# hp.mollview(emission[0], norm='hist')
# plt.show()
# frequency_map = sky.full_sky(nus*u.GHz, bandpass1*u.K)
# hp.mollview(frequency_map[0], norm='hist')
# frequency_map = sky.full_sky(22*u.GHz)
# hp.mollview(frequency_map[0], norm='hist')
# plt.show()

freqs, rms = sky.spectrum(sky_frac=50)
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(5e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [uK_RJ]')
plt.legend()
plt.show()
