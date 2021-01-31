"""
For testing Cosmoglobe during development.

"""

from healpy.pixelfunc import remove_dipole, remove_monopole
from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import matplotlib.pyplot as plt
import pathlib
import healpy as hp
import numpy as np
import astropy.units as u

# data = pathlib.Path('../../Cosmoglobe_test_data/reduced_chain_test.h5')
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
data = '../../Cosmoglobe_test_data/chain_test.h5'

cosmo = Cosmoglobe(data, sample='mean')

cmb = cosmo.model('cmb', remove_dipole=True, remove_monopole=True)
synch = cosmo.model('synch')
dust = cosmo.model('dust')
ff = cosmo.model('ff')
ame = cosmo.model('ame')

# sky_500GHz = cosmo.full_sky(nu=500*u.GHz)
# hp.mollview(sky_500GHz[0], min=0, max=1000)
# plt.show()

freqs, rms = cosmo.spectrum()
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(1e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()

