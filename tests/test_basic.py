"""
For testing Cosmoglobe during development.

"""
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from context import cosmoglobe
from cosmoglobe.tools.map import to_IQU
import cosmoglobe.plot as cgp
path = "/Users/svalheim/work/cosmoglobe-workdir/"
hp.disable_warnings()
map_=hp.read_map(path+"cmb_c0001_k000200.fits")

mask = np.random.randint(2, size=len(map_))
cgp.mollplot(map_, colorbar=True)
plt.show()
