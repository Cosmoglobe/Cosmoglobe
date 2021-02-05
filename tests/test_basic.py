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

# bandpass = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
# freqs, det1, det2 = np.loadtxt(bandpass, unpack=True)

# bandpass = '../../Cosmoglobe_test_data/wmap_bandpass.txt'

data = '../../Cosmoglobe_test_data/reduced_chain_test.h5'
sky = Cosmoglobe(data, sample='mean')
# cmb = sky.model('cmb')
cmb = sky.model('cmb', remove_dipole=True, remove_monopole=True)
synch = sky.model('synch')
dust = sky.model('dust')
ff = sky.model('ff')
ame = sky.model('ame')

# frequency_map = sky.full_sky(nu=30*u.GHz)
# hp.mollview(frequency_map[0], min=-3400, max=3400)

# map_ = synch.get_emission(freqs*u.GHz, det1*(u.Jy/u.sr))
# print(synch.get_emission(66*u.GHz))
freqs, rms = sky.spectrum(sky_frac=22)
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(5e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()

