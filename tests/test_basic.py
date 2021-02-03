"""
For testing Cosmoglobe during development.

"""

from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe
import matplotlib.pyplot as plt
# from cmcrameri import cm
import pathlib
import healpy as hp
import numpy as np
import astropy.units as u

hp.disable_warnings()
data = '../../Cosmoglobe_test_data/reduced_chain_test.h5'
# bandpass = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
# freqs, det1, det2 = np.loadtxt(bandpass, unpack=True)
# a = 1*u.K
# print(a.to(u.Jy/u.sr, equivalencies=u.thermodynamic_temperature(30*u.GHz)).value)
# data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')
# data = '../../Cosmoglobe_test_data/chain_test.h5'

cosmo = Cosmoglobe(data, sample='mean')
# cosmo.reduce_chainfile()
cmb = cosmo.model('cmb', remove_dipole=True, remove_monopole=True)
# hp.mollview(cmb.amp[0], min=-300, max=300, cmap=cm.batlow)
# plt.show()
synch = cosmo.model('synch')
dust = cosmo.model('dust')
ff = cosmo.model('ff')
ame = cosmo.model('ame')
# freqs = np.arange(1,10,1)
# weights = np.ones(len(freqs), dtype=np.float)
# weights = weights/np.trapz(weights, freqs)
# weights = weights/np.trapz(weights, freqs)

# print(weights)
# plt.plot(freqs, weights)
# plt.show()



# synch[nus]



freqs, rms = cosmo.spectrum()


for model, model_rms in rms.items():
    plt.loglog(freqs, model_rms, label=model)

plt.ylim(1e-2, 1e3)
plt.xlabel('Frequency [GHz]')
plt.ylabel('RMS brightness temperature [muK]')
plt.legend()
plt.show()

# frequency_map = cosmo.full_sky(nu=1000*u.GHz)
# hp.mollview(frequency_map[0], min=0, max=2000)
# plt.show()

