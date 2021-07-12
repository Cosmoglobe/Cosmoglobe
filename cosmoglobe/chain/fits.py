from cosmoglobe.sky import model_from_chain
from cosmoglobe.sky.model import Model
from cosmoglobe.hub import COSMOGLOBE_COMPS

from astropy.io import fits
import astropy.units as u
import pathlib
import numpy as np


def chain_to_fits(chainfile, dirname, burn_in=None):
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

    for nside in DEFAULT_NSIDES:
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
            if not isinstance(freq_ref, np.ndarray):
                freq_ref = np.asarray(freq_ref)
            
            freq_hdu = fits.ImageHDU(
                data=freq_ref, 
                name=f'{component.label}_freq',
            )
            hdu_list.append(freq_hdu)

        for key, value in component.spectral_parameters.items():
            spectral_hdu = fits.ImageHDU(
                data=value.value,
                name=f'{component.label}_{key}',
            )
            hdu_list.append(spectral_hdu)
        
    hdu_list.writeto(filename)


def model_from_fits(dirname, nside):
    """Initializes a model from a fits file.

    """

    dirname = pathlib.Path(dirname)
    filename = dirname / f'model_{nside}.fits'

    with fits.open(filename) as hdu_list:
        model_dict = dict.fromkeys(COSMOGLOBE_COMPS.keys(), {})
        for hdu in hdu_list[1:]:
            comp, item = hdu.name.split("_")
            if item.lower() == 'amp':
                # value = 
                pass


            # model_dict[hdu.name]['amp'] = 

            # amp_unit = u.mK if comp == 'radio' else u.uK
            # model_dict[comp] = {}
            # model_dict[comp]['amp'] = u.Quantity(
            #     hdu_list[f'{comp}_amp'].data, unit=amp_unit
            # )

            # model_dict[comp]['spectral_parameters'] = {}


            

        model = Model(nside=nside)







def model_from_dict(model_dict):
    """Creates and returns a cosmoglobe.sky.Model from a model dictionary."""
    model = Model()
    for key, value in model_dict.items():
        try:
            comp = COSMOGLOBE_COMPS[key]
        except KeyError:
            raise ModelError(
                'Model components does not match the Cosmoglobe Sky Model'
            )

        try:
            amp = value['amp']
            freq_ref = value['freq_ref']
            spectral_parameters = value['spectral_parameters']
        except KeyError as e:
            raise e

        if freq_ref is not None: #CMB is initialized without a reference freq
            model.insert(comp(key, amp, freq_ref, **spectral_parameters))    
        else:
            model.insert(comp(key, amp, **spectral_parameters))    
        
    return model