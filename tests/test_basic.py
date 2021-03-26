"""
For testing Cosmoglobe during development.

"""
path = "/Users/svalheim/work/cosmoglobe-workdir/"
from context import cosmoglobe
import cosmoglobe.plot as cgp
import healpy as hp

map = hp.read_map(path+"cmb_c0001_k000200.fits")

cgp.mollplot(map)