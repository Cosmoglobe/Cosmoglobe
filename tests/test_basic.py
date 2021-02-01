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
data = '../../Cosmoglobe_test_data/reduced_chain_test.h5'
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
# data = '../../Cosmoglobe_test_data/chain_test.h5'

cosmo = Cosmoglobe(data, sample='mean')
# cosmo.reduce_chainfile()
cmb = cosmo.model('cmb', remove_monopole=True, remove_dipole=True)
synch = cosmo.model('synch')
dust = cosmo.model('dust')
ff = cosmo.model('ff')
ame = cosmo.model('ame')


freqs, rms = cosmo.spectrum()
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(1e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()

# frequency_map = cosmo.full_sky(nu=70*u.GHz, models=[ff, ame])
# hp.mollview(synch[30*u.GHz][0], min=50, max=400, unit=synch.params['unit'], title=synch.params['type'])
# plt.show()