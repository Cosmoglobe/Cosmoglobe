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
data = '../../Cosmoglobe_test_data/bla.h5'


from cosmoglobe.sky.map import IQUMap, to_IQU
import healpy as hp
from cosmoglobe.sky.components import PowerLaw, ModifiedBlackBody
from cosmoglobe.sky.model import Model
map_path = '/Users/metinsan/Documents/doktor/Cosmoglobe/cosmoglobe/data/BP7_70GHz_nocmb_n0256.fits'

# print(sky.synch._get_freq_scaling((50*u.GHz).value, sky.synch.beta))

synch = PowerLaw(sky.synch.amp, sky.synch.params.nu_ref, beta=sky.synch.beta)
dust = ModifiedBlackBody(sky.dust.amp, sky.dust.params.nu_ref, beta=sky.dust.beta, T=sky.dust.T)
# print(synch.__class__.__name__)

model = Model(components=[('synch', synch), ('dust', dust)])


print(synch.amp)
print(synch.amp.I)
print(synch.amp.unit)



