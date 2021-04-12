"""For testing Cosmoglobe during development."""
from context import cosmoglobe
import cosmoglobe.sky as sky
import astropy.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

hp.disable_warnings()

data = '../../Cosmoglobe_test_data/bla.h5'
map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'
bandpass_data = '../../Cosmoglobe_test_data/wmap_bandpass.txt'
nu_array, bandpass_array, _ = np.loadtxt(bandpass_data, unpack=True)

nside = 512
model = sky.model_from_chain(data, nside=nside, sample=20)
# model.to_nside(16)
# print('16',hp.nside2npix(16))
# print('256',hp.nside2npix(256))
# for comp in model:
#     print(comp.amp.shape)
#     for key, value in comp.spectral_parameters.items():
#         print(comp.name, key, np.shape(value))

# model.dust.spectral_parameters['T'] = i2*u.K
# model.dust.spectral_parameters['beta'] = i2*u.dimensionless_unscaled
# print(model)
# for comp in model:
#     print(comp.name, comp.amp.unit, comp.get_emission(150*u.GHz))
# print((model.ff.amp).shape)
# print((model.dust.amp).shape)
# print((model.ame.amp).shape)
# print((model.cmb.amp).shape)
# print((model.synch.amp).shape)

# print((model.ff.get_emission(400*u.GHz)).shape)
# print((model.dust.get_emission(400*u.GHz)).shape)
# print((model.ame.get_emission(400*u.GHz)).shape)
# print((model.cmb.get_emission(400*u.GHz)).shape)
# print((model.synch.get_emission(400*u.GHz)).shape)
# print(model.dust.spectral_parameters)
# for comp in model:
    # for idx, col in enumerate(comp.amp):
        # hp.mollview(col, norm='hist', title=f'{comp.name} {idx}')


# emission = model.get_emission(nu_array*u.GHz, bandpass_array*u.uK)
# emission = model.get_emission(nu_array*u.GHz, bandpass_array*u.uK, output_unit=(u.MJy/u.sr))
# print(emission.unit)
# hp.mollview(emission[0], norm='hist', title="bp I")
# hp.mollview(emission[1], norm='hist', title="bp Q")
# hp.mollview(emission[2], norm='hist', title="bp U")

# emission = model.get_emission(150*u.GHz)
# emission = model.get_emission(150*u.GHz, output_unit=(u.MJy/u.sr))
# hp.mollview(emission[0], norm='hist', title="150 I")
# hp.mollview(emission[1], norm='hist', title="150 Q")
# hp.mollview(emission[2], norm='hist', title="150 U")

# plt.show()
