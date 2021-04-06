"""
For testing Cosmoglobe during development.
"""
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from context import cosmoglobe

from cosmoglobe.tools.map import to_stokes
import cosmoglobe.plot as cgp
path = "/Users/svalheim/work/cosmoglobe-workdir/"
hp.disable_warnings()
map_=hp.read_map(path+"cmb_c0001_k000200.fits", field=None)
map_ = to_stokes(map_, label="cmb")

#mask = np.random.randint(2, size=len(map_))
mask=map_.data>0
mask[1] = mask[0]
cgp.mollplot(map_, colorbar=True, mdmask="auto")
#plt.show()
