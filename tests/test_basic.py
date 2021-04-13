"""
For testing Cosmoglobe during development.
"""
import astropy.units as u
import healpy as hp
import numpy as np
import healpy as hp
from context import cosmoglobe
import matplotlib.pyplot as plt
from cosmoglobe.tools.map import to_stokes
import cosmoglobe.plot as cgp
path = "/Users/svalheim/work/cosmoglobe-workdir/"
hp.disable_warnings()
map_=hp.read_map(path+"cmb_c0001_k000200.fits", field=None)

mask=map_>0
mask[1] = mask[0]

cgp.mollplot(map_, auto="cmb",)
#cgp.gnomplot(map_, 0,-70, auto="cmb", remove_dip=True, subplot=(1,2,1))
#cgp.gnomplot(map_, 0,-70, auto="cmb", sig=1, fwhm=30, subplot=(1,2,2))
plt.show()
