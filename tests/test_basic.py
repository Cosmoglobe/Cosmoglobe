"""
For testing Cosmoglobe during development.

"""


from healpy.pixelfunc import remove_monopole
from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import matplotlib.pyplot as plt
import pathlib
import healpy as hp
import numpy as np
import astropy.units as u

data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')

sky = Cosmoglobe(data)

cmb = sky.model('cmb', remove_dipole=True, remove_monopole=True)
# cmb.to_nside(32)
# hp.mollview(cmb.amp[0], min=-300, max=300)
# plt.show()
dust = sky.model('dust')
synch = sky.model('synch')
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
