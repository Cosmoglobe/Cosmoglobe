# Cosmoglobe Sky Model
Create full-sky simulations directly from Commander3 outputs or from other configurations. *(The project is still in beta version)*

## Installation
```bash
pip install cosmoglobe
```

## Example Usage
The documentation for the project is still under development. Following is 
an example of how the package can be used to produce sky simulations.

```python
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from cosmoglobe import SkyModel

hp.disable_warnings()

data_from_chain = 'path/to/chain.h5'
# data_from_config = 'path/to/custom_config.json'

bandpass_data = 'path/to/bandpass.txt'
bandpass_freqs, bandpass_profile = np.loadtxt(bandpass_data, unpack=True)

sky = SkyModel(data_from_chain, components=['synch', 'dust'], nside=512, 
               fwhm=150*u.arcmin, sample='mean', burn_in=15)

# sky = SkyModel(data_from_json)

# print(sky.synch.params.nu_ref)
# synch_ref_amp = sky.synch.amp
# dust_50GHz = sky.dust.get_emission(50*u.GHz)
# full_sky_300GHz = sky.get_emission(300*u.GHz)

# dust_bp_integrated = sky.dust.get_emission(bandpass_freqs*u.GHz, 
#                                            bandpass_profile*u.K)

full_sky_bp_integrated = sky.get_emission(bandpass_freqs*u.GHz, 
                                          bandpass_profile*u.K)

hp.mollview(full_sky_bp_integrated[0], norm='hist')
plt.show()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)