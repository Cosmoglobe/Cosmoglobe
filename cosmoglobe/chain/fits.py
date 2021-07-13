from cosmoglobe.sky import model_from_chain
from cosmoglobe.sky.model import Model
from cosmoglobe.hub import COSMOGLOBE_COMPS

from astropy.io import fits
import astropy.units as u
import pathlib
import numpy as np


def chain_to_fits(chainfile, dirname, nside=None, burn_in=None):
    """Writes a commander3 chain to a sky model fits file.
    Parameters
    ----------
    chainfile : str, `pathlib.PosixPath`
        Path to chain.
    filename : str, `pathlib.PosixPath`
        Path to output fits file.
    burn_in : int
        The sample number as a str or int where the chainfile is assumed to 
        have sufficently burned in. All samples before the burn_in are ignored 
        in the averaging process.
    """

    DEFAULT_NSIDES = [
        1,
        2,
        4,
        8,
        # 16,
        # 32,
        # 64,
        # 128,
        # 256,
        # 512,
        # 1024,
        # 2048,
    ]

    nsides = nside if not nside is None else DEFAULT_NSIDES

    for nside in nsides:
        model = model_from_chain(chainfile, nside=nside, burn_in=burn_in)
        model_to_fits(model, dirname)



def model_to_fits(model, dirname):
    """Writes a `cosmoglobe.sky.Model` to a fits file.

    Parameters
    ----------
    model : `cosmoglobe.sky.Model`
        A cosmoglobe sky model.
    dirname : str, `pathlib.PosixPath`
        dirname.
    """

    dirname = pathlib.Path(dirname)
    dirname.mkdir(parents=True, exist_ok=True)
    filename = dirname / f'model_{model.nside}.fits'

    hdu_list = fits.HDUList()
    primary_hdu = fits.PrimaryHDU()
    hdu_list.insert(0, primary_hdu)
    for component in model:
        amp_hdu = fits.ImageHDU(
            data=component.amp.value, 
            name=f'{component.label}_amp',
        )
        hdu_list.append(amp_hdu)

        if component.freq_ref is not None:
            freq_ref = component.freq_ref.value
            if not isinstance(freq_ref, u.Quantity):
                freq_ref = np.array([freq_ref])
            
            freq_hdu = fits.ImageHDU(
                data=freq_ref, 
                name=f'{component.label}_freq',
            )
            hdu_list.append(freq_hdu)

        for key, value in component.spectral_parameters.items():
            spectral_hdu = fits.ImageHDU(
                data=value.value,
                name=f'{component.label}_sp_{key}',
            )
            hdu_list.append(spectral_hdu)
        
    hdu_list.writeto(filename)


def model_from_fits(dirname, nside):
    """Initializes a model from a fits file.

    """

    dirname = pathlib.Path(dirname)
    filename = dirname / f'model_{nside}.fits'

    with fits.open(filename) as hdu_list:
        model_dict = dict.fromkeys(
            COSMOGLOBE_COMPS.keys(), 
            {'spectral_parameters':{}}
        )
        for hdu in hdu_list[1:]:
            comp, item = hdu.name.lower().split("_", 1)
            print(item)
            if item == 'amp':
                unit = u.uK if comp != 'radio' else u.mK

            elif item in ['freq', 'nu_p']:
                unit = u.GHz

            elif item in ['T', 'Te']:
                unit = u.K

            else:
                unit = u.dimensionless_unscaled

            value = hdu.data * unit

            # if '

            if item == 'amp':
                model_dict[comp]['amp'] = value
            elif item == 'freq':
                model_dict[comp]['freq_ref'] = value
            else:
                model_dict[comp]['spectral_parameters'][item] = value            

        model = model_from_dict(model_dict)
        print(model)







def model_from_dict(model_dict):
    """Creates and returns a cosmoglobe.sky.Model from a model dictionary."""
    model = Model()
    for key, value in model_dict.items():
        try:
            comp = COSMOGLOBE_COMPS[key]
        except KeyError:
            raise KeyError(
                'Model components does not match the Cosmoglobe Sky Model'
            )

        try:
            amp = value['amp']
            freq_ref = value['freq_ref']
            spectral_parameters = value['spectral_parameters']
        except KeyError as e:
            raise e

        # if freq_ref is not None: #CMB is initialized without a reference freq
        print(comp, amp.shape, freq_ref.shape, spectral_parameters.keys())
        model._insert_component(comp(amp, freq_ref, **spectral_parameters))    
        # else:
            # model._insert_component(comp(amp, **spectral_parameters))    
        
    return model