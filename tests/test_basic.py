"""
For testing Cosmoglobe during development.
"""
import astropy.units as u
import healpy as hp
import numpy as np
import healpy as hp
from context import cosmoglobe
import matplotlib.pyplot as plt
import cosmoglobe.plot as cgp
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
"""
N = 1000
input = np.zeros((2,N))
input[0] =  np.random.normal(-3.1,0.1, N)
input[1] =  np.random.normal(-2.8,0.1, N)
cgp.traceplot(input, header=["first", "second"], labelval=True, subplot=(2,1,1))
cgp.traceplot(input+0.1, header=["third", "forth"], labelval=True, subplot=(2,1,2))
plt.show()
"""
chainfile="/mn/stornext/u3/trygvels/compsep/cdata/like/commander_workdirs/BP8/chains_leak1/chain_c0001.h5"

#cgp.traceplot(input, header=["first", "second"], labelval=True, subplot=(2,1,1))
components = {"bandpass": ["030", "044", "070"], "synch": ["beta_pixreg_val"], "dust": ["beta_alm",], "ame": ["nu_p_pixreg_val"] }

from cosmoglobe.tools.h5 import _get_samples, _get_items


i=1
for comp in components.keys():
    for value in components[comp]:
        label = comp+"-"+value
        dat = []
        for sample in _get_samples(chainfile):
            dat.append(_get_items(chainfile, sample, comp, value))
        
        dat = np.array(dat).T
        if "alm" in value:
            dat /= np.sqrt(4*np.pi)

        for p in range(dat.shape[1]):
            cgp.traceplot(dat[:,p,:],ylabel=label, labelval=True, figsize=(12,4),  subplot=(6,1,i))

        i+=1


plt.savefig("trace.png")


