"""
For testing Cosmoglobe during development.

"""

from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import pathlib

data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')

sky = Cosmoglobe(data)
model = sky.model('ff')
model_200 = model[200*u.GHz]

hp.mollview(model_200[0], title=model.params['type'], 
            norm='hist', unit=model.params['unit'])
            
print(model)            
print(repr(model))
print(model.model_name)
print(model.params)            

plt.show()

