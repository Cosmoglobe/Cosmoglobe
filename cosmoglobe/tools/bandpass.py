import numpy as np
import astropy.units as u
from scipy.interpolate import RectBivariateSpline


@u.quantity_input(bandpass=(u.Jy/u.sr, u.K), freqs=u.Hz)
def _get_normalized_bandpass(bandpass, freqs, input_unit):
    """Normalize bandpass to integrate to unity under numpy trapzoidal"""
    bandpass = bandpass.to(
        input_unit, equivalencies=u.brightness_temperature(freqs)
    )
    return bandpass/np.trapz(bandpass, freqs)


@u.quantity_input(bandpass=(u.Jy/u.sr, u.K), freqs=u.Hz)
def _get_unit_conversion(bandpass, freqs, output_unit, input_unit):
    """Unit conversion factor given a input and output unit"""
    bandpass_output = bandpass.to(
        output_unit, equivalencies=u.brightness_temperature(freqs)
    )
    bandpass_input = bandpass.to(
        input_unit, equivalencies=u.brightness_temperature(freqs)
    )
    return np.trapz(bandpass_output, freqs)/np.trapz(bandpass_input, freqs)


def _get_interp_parameters(spectral_parameters, n=10):
    interp_parameters = {}
    for key, value in spectral_parameters.items():
        if value.ndim > 1:
            if value.shape[1] > 1:
                min_, max_ = np.amin(value), np.amax(value)
                interp = np.linspace(min_, max_, n)
                interp_parameters[key] = interp

    return interp_parameters


def _interp1d(comp, bandpass, freqs, freq_ref, interp_parameter, 
              spectral_parameters):
    integrals = []
    for key, interp_param in interp_parameter.items():
        for grid_point in interp_param:
            spectral_parameters[key] = grid_point
            freq_scaling = comp.get_freq_scaling(freqs, freq_ref, 
                                                 **spectral_parameters)
            integrals.append(np.trapz(freq_scaling*bandpass, freqs))

        #convert shape to (3, n) 
        integrals = u.Quantity(np.transpose(integrals), unit=integrals[0].unit)
        if integrals.ndim == 1:
            integrals = np.expand_dims(integrals, axis=0)

        if len(comp.spectral_parameters[key]) == 3:
            scaling =  [np.interp(comp.spectral_parameters[key][idx].value, 
                                  interp_param.value, 
                                  col.value)
                        for idx, col in enumerate(integrals)]
        else:
            scaling =  [np.interp(comp.spectral_parameters[key].value, 
                                  interp_param.value, 
                                  col.value)
                        for col in integrals]
        return scaling


def _interp2d(comp, bandpass, freqs, freq_ref, interp_parameters,
              spectral_parameters):
    n = len(list(interp_parameters.values())[0])
    # Square n x n grid for each spectral_parameter
    grid = {
        key: value for key, value in zip(
            interp_parameters.keys(), np.meshgrid(*interp_parameters.values())
        )
    }

    if comp._is_polarized:
        integrals = np.zeros((n, n, 3))
    else:
        integrals = np.zeros((n, n, 1))
    for i in range(n):
        for j in range(n):
            grid_spectrals = {key: value[i,j] for key, value in grid.items()}
            freq_scaling = comp.get_freq_scaling(freqs, freq_ref, 
                                                 **grid_spectrals)
            integrals[i,j] = np.trapz(freq_scaling*bandpass, freqs)
    integrals = np.transpose(integrals)

    scaling = []
    for idx, col in enumerate(integrals):
        f = RectBivariateSpline(*interp_parameters.values(), col)
        packed_spectrals = [spectral[idx] if spectral.shape[0] == 3 else spectral[0]
                            for spectral in spectral_parameters.values()]
        scaling.append(
            f(*packed_spectrals, grid=False)
        )
    
    return scaling