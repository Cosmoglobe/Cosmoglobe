from ..tools import utils

import numpy as np
import astropy.units as u

import sys

def _normalize_bandpass(bandpass: u.Quantity, freqs: u.Quantity) -> u.Quantity:
    if bandpass.si.unit != u.K:
        bandpass = bandpass.to(u.K, equivalencies=u.brightness_temperature(freqs))

    return bandpass / np.trapz(bandpass.si.value, freqs.si.value)


def _get_interp_params(spectral_parameters):

        print(spectral_parameters)
        interpolation_parameters = {}
        for key, value in spectral_parameters.items():
            min_, max_ = np.amin(value), np.amax(value)
            print(min_, max_)

        # interp_ranges = {}

        # for key, value in self.spectral_parameters.items():
        #     if scalars is None or key not in scalars:
        #         interp_ranges[key] = np.linspace(np.amin(value), 
        #                                          np.amax(value), 
        #                                          n) * value.unit


                



    # if isinstance(iterator, (tuple, list)):
    #     scalars = []
    #     for value in iterator:
    #         uniques = np.unique(value.data)
    #         if len(uniques) == 1:
    #             scalar = uniques[0]
    #             scalars.append(scalar)
        
    #     if scalars:
    #         if isinstance(iterator, tuple):
    #             return tuple(scalars)

    #         return scalars
    
    # if isinstance(iterator, dict):
    #     scalars = {}
    #     for key, value in iterator.items():
    #         uniques = np.unique(value.data)
    #         if len(uniques) == 1:
    #             scalar = uniques[0]
    #             scalars[key] = scalar

    #     if scalars:
    #         return scalars

    # return


