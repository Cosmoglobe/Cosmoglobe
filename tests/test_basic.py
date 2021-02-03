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

data = '../../Cosmoglobe_test_data/reduced_chain_test.h5'
bandpass = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
# data = '../../Cosmoglobe_test_data/chain_test.h5'

cosmo = Cosmoglobe(data, sample='mean')
# cosmo.reduce_chainfile()

cmb = cosmo.model('cmb', remove_dipole=True, remove_monopole=True)
synch = cosmo.model('synch')
dust = cosmo.model('dust')
ff = cosmo.model('ff')
ame = cosmo.model('ame')


# frequency_map = cosmo.full_sky(nu=1000*u.GHz)
# hp.mollview(frequency_map[0], min=0, max=2000)

# freqs, rms = cosmo.spectrum()
# for model, model_rms in rms.items():
#     plt.loglog(freqs, model_rms, label=model)

# plt.ylim(1e-2, 1e3)
# plt.xlabel('Frequency [GHz]')
# plt.ylabel('RMS brightness temperature [muK]')
# plt.legend()

# plt.show()

