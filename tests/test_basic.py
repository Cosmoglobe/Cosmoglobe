"""
For testing Cosmoglobe during development.

"""


from healpy.pixelfunc import remove_dipole
from context import cosmoglobe
from cosmoglobe.cosmoglobe import Cosmoglobe

import matplotlib.pyplot as plt
import pathlib
import healpy as hp
import numpy as np

data = pathlib.Path('../../Cosmoglobe_test_data/chain_test.h5')

sky = Cosmoglobe(data)
model = sky.model('cmb', remove_monopole=True)
# model2 = sky.model('cmb')


# map_mono = model2.amp[0].value
# map_ = model1.amp[0].value

# model = sky.model('cmb', remove_dipole=True)
# model = sky.model('cmb')
# model = sky.model('cmb')

# np.save('../../Cosmoglobe_test_data/cmb_mono.npy', model.amp[0].value)
# mono = np.load('../../Cosmoglobe_test_data/cmb_mono.npy')
# hp.mollview(model.amp[0], title=model.params['type'], 
#             norm='hist', unit=model.params['unit'])
# hp.mollview(model.amp[0])
hp.mollview(model.amp[0], norm='hist')

plt.show()
