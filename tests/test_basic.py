"""
For testing Cosmoglobe during development.

"""
path = "/Users/svalheim/work/cosmoglobe-workdir/"
from context import cosmoglobe
import cosmoglobe.plot as cgp
from cosmoglobe.tools.map import *
import healpy as hp
import matplotlib.pyplot as plt

map_ = hp.read_map(path+"cmb_c0001_k000200.fits")
map_ = to_IQU(map_)
map_.label = "cmb"

mask = np.random.randint(2, size=len(map_.I))
cgp.mollplot(map_, colorbar=True)
plt.show()

