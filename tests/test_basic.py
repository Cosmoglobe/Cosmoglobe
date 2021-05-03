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
"""
map_=hp.read_map(path+"cmb_c0001_k000200.fits", field=None)

mask=map_>0
mask[1] = mask[0]

cgp.mollplot(map_, auto="cmb", remove_dip=True, subplot=(1,2,1))
cgp.gnomplot(map_, 0,-70, auto="cmb", remove_dip=True, subplot=(1,2,2))
"""
#cgp.gnomplot(map_, 0,-70, auto="cmb", sig=1, fwhm=30, subplot=(1,2,2))

#_, bins, _ = cgp.hist(x, bins=50, prior=(-3.1,0.1), label="lol")
#x = np.random.normal(-2.8,0.1, 1000)
#cgp.hist(x, bins=50, prior=(-2.8,0.1), label="lol2")
#plt.legend(frameon=False, loc="upper right")
N = 1000
input = np.zeros((2,N))
input[0] =  np.random.normal(-3.1,0.1, N)
input[1] =  np.random.normal(-2.8,0.1, N)
cgp.traceplot(input, header=["first", "second"], labelval=True, subplot=(2,1,1))
cgp.traceplot(input+0.1, header=["third", "forth"], labelval=True, subplot=(2,1,2))
plt.show()

"""
Stort plot
Traceplot:
chisq
Dust beta
dust temp
synch beta
nu_p

chisq_c0001_k000029.fits
cmb_c0001_k000029.fits
  res_0.4-Haslam_c0001_k000029.fits
  res_030-WMAP_Ka_c0001_k000029.fits
  res_030_c0001_k000029.fits
  res_033-WMAP_Ka_P_c0001_k000029.fits
  res_040-WMAP_Q1_c0001_k000029.fits
  res_040-WMAP_Q2_c0001_k000029.fits
  res_041-WMAP_Q_P_c0001_k000029.fits
  res_044_c0001_k000029.fits
  res_060-WMAP_V1_c0001_k000029.fits
  res_060-WMAP_V2_c0001_k000029.fits
  res_061-WMAP_V_P_c0001_k000029.fits
  res_070_c0001_k000029.fits
  res_353_c0001_k000029.fits
  res_857_c0001_k000029.fits
"""