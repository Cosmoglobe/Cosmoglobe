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

# data = pathlib.Path('../../Cosmoglobe_test_data/reduced_chain_test.h5')
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
data = '../../Cosmoglobe_test_data/chain_test.h5'

sky = Cosmoglobe(data, sample='mean')
cmb = sky.model('cmb', remove_dipole=True, remove_monopole=True)
synch = sky.model('synch')
dust = sky.model('dust')
ff = sky.model('ff')
ame = sky.model('ame')

freqs, rms = sky.spectrum()
for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(1e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()
