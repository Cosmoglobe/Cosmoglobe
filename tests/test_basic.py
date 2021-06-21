from context import cosmoglobe
from cosmoglobe.sky import model_from_chain

import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
hp.disable_warnings()
chain = "/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/bla.h5"

model = model_from_chain(chain, nside=32, burn_in=20)
print(model.dust.freq_ref)
hp.mollview(model(353*u.GHz)[0], norm='hist')
plt.show()