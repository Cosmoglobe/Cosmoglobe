"""
For testing Cosmoglobe during development.

"""

import sys
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

hp.disable_warnings()

data = '../../Cosmoglobe_test_data/chain_test.h5'
sky = Cosmoglobe(data, sample='mean')
# cmb = sky.model('cmb')
cmb = sky.model('cmb', remove_dipole=True, remove_monopole=True)
synch = sky.model('synch')
dust = sky.model('dust')
ff = sky.model('ff')
ame = sky.model('ame')

# dust.to_nside(64)
# dust.smooth(150*u.arcmin)
# hp.mollview(dust.amp[0], norm='hist')
# plt.show()
#
# frequency_map = sky.full_sky(nu=30*u.GHz)
# hp.mollview(frequency_map[0], min=-3400, max=3400)

freqs, rms = sky.spectrum(sky_frac=50)
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(5e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [uK_RJ]')
plt.legend()
plt.show()

