from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt

sky = Cosmoglobe('../cosmoglobe/data/chain_test.h5')
# print(sky)
model = sky.model('AmE')

# print(model.model_name)

# print(model.params)
model_shifted = model[900*u.GHz]
# print(model_shifted)
# print(model.beta[0], model.beta[1], model.beta[2])
hp.mollview(model.amp[0], title=model.params['type'], 
            norm='hist', unit=model.params['unit'])
            
hp.mollview(model_shifted[0], title=model.params['type'], 
            norm='hist', unit=model.params['unit'])

plt.show()
