"""
For testing Cosmoglobe during development.

"""
# from cosmoglobe.sky import components
import astropy.units as u
import healpy as hp
from healpy.pixelfunc import nside2npix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import imageio
import os
import h5py

from context import cosmoglobe
from cosmoglobe.cosmoglobe import SkyModel

hp.disable_warnings()
# planck_cmap = ListedColormap(np.loadtxt('../../Cosmoglobe_test_data/planck_cmap.dat')/255.)

data = '../../Cosmoglobe_test_data/bla.h5'
# data = '../../Cosmoglobe_test_data/data.json'

# bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
# nus, bandpass, _ = np.loadtxt(bandpass_data, unpack=True)
sky = SkyModel(data, components=['synch', 'dust'], nside=64, burn_in=20)

# data = h5py.File('../../Cosmoglobe_test_data/bla.h5', 'r')
# beta_map = data['000030/synch/beta_map']
# hp.mollview(beta_map[1])

# hp.mollview(sky.dust.amp[0], min=-50, max=50)
# hp.mollview(sky.get_emission(nus*u.GHz, bandpass*u.K)[0], norm='hist')
# plt.show()

# freqs = np.arange(1,300,3)
# sky_list = [sky.get_emission(freq*u.GHz)[0] for freq in freqs]
# filenames = []
# for freq in freqs:
#     hp.mollview(sky.get_emission(freq*u.GHz)[0], norm='hist', title=f'Sky at {freq} GHz')
#     filename = f'gif/{freq}.png'
#     filenames.append(filename)
#     plt.savefig(filename)
#     plt.close()


# with imageio.get_writer('sky.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
        
# for filename in set(filenames):
#     os.remove(filename)

# plt.show()
# hp.write_map('../../Cosmoglobe_test_data/ff_amp_n1024.fits', sky.ff.amp)
# hp.write_map('../../Cosmoglobe_test_data/ff_Te_map_n1024.fits', sky.ff.Te_map)

# plt.show()

# freqs, rms = sky.get_spectrum(sky_frac=50,pol=True)
# for model, model_rms in rms.items():
#     plt.loglog(freqs, model_rms, label=model)

# plt.ylim(5e-2, 1e3)
# plt.xlabel('Frequency [GHz]')
# plt.ylabel('RMS brightness temperature [uK_RJ]')
# plt.legend()
# plt.show()

from cosmoglobe.sky.map import IQUMap, to_IQU
import healpy as hp
from cosmoglobe.sky.components import PowerLaw, ModifiedBlackBody
from cosmoglobe.sky.model import Model
map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'

# print(sky.synch._get_freq_scaling((50*u.GHz).value, sky.synch.beta))

synch = PowerLaw(sky.synch.amp, sky.synch.params.nu_ref, beta=sky.synch.beta)
dust = ModifiedBlackBody(sky.dust.amp, sky.dust.params.nu_ref, beta=sky.dust.beta, T=sky.dust.T)
# print(synch.__class__.__name__)

model = Model(components=[('synch', synch)])

print(model)
