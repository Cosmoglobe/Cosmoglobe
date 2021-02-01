"""
For testing Cosmoglobe during development.

"""
from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import matplotlib.pyplot as plt
import pathlib
import healpy as hp
import numpy as np
import astropy.units as u

hp.disable_warnings()
# data = '../../Cosmoglobe_test_data/reduced_chain_test.h5'
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
data = '../../Cosmoglobe_test_data/chain_test.h5'

cosmo = Cosmoglobe(data, sample='mean', burnin=1)

cmb = cosmo.model('cmb', remove_monopole=True, remove_dipole=True)
synch = cosmo.model('synch')
dust = cosmo.model('dust')
ff = cosmo.model('ff')
ame = cosmo.model('ame')

# frequency_map_30GHz = cosmo.full_sky(nu=30*u.GHz)
# hp.mollview(frequency_map_30GHz[0])
# plt.show()

freqs, rms = cosmo.spectrum()
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(1e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()
