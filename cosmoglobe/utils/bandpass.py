from .constants import h, c, k_B, T_0

from scipy.interpolate import RectBivariateSpline
import astropy.units as u
import numpy as np

@u.quantity_input(bandpass=(u.Jy/u.sr, u.K), freqs=u.Hz)
def get_normalized_bandpass(bandpass, freqs, unit=u.K):
    """Returns a bandpass profile normalized to unity when integrated over all 
    frequencies corresponding to the profile in units of K_RJ.

    Parameters
    ----------
    bandpass: `astropy.units.Quantity`
        Bandpass profile in signal units.    
    freqs: `astropy.units.Quantity`
        Frequencies corresponding to the bandpass profile.
    unit: `astropy.units.UnitBase`
        The unit we want to convert to.
    """
    if bandpass.unit == (u.MJy/u.sr) or bandpass.unit == (u.Jy/u.sr):
        bandpass = bandpass.to(
            unit=unit, equivalencies=u.brightness_temperature(freqs)
        )

    return bandpass/np.trapz(bandpass, freqs)


@u.quantity_input(bandpass=u.Hz**-1, freqs=u.Hz)
def get_bandpass_coefficient(bandpass, freqs, output_unit):
    """Computes the bandpass coefficient for a map that has been calibrated in 
    K_RJ to some output unit, after taking into account the bandpass.

    Parameters
    ----------
    bandpass: `astropy.units.Quantity`
        Normalized bandpass profile in units of 1/Hz.    
    freqs: `astropy.units.Quantity`
        Frequencies corresponding to the bandpass profile.
    unit: `astropy.units.UnitBase`
        The unit we want to convert to.

    Returns
    -------
    bandpass_coefficient: `astropy.units.Quantity`
        The bandpass coefficient.
        
    """
    intensity_derivative_i = b_rj(freqs)

    #Selecting the intensity derivative expression given an output_unit
    try:
        output_unit = u.Unit(output_unit)
        if u.K in output_unit.si.bases:
            intensity_derivative_j = b_rj(freqs)
        elif output_unit == u.MJy/u.sr:
            intensity_derivative_j = b_iras(freqs, np.mean(freqs))

    except ValueError:
        if isinstance(output_unit, str):
            if output_unit.lower().endswith('k_rj'):
                intensity_derivative_j = b_rj(freqs)
            elif output_unit.lower().endswith('k_cmb'):
                intensity_derivative_j = b_cmb(freqs)
            else:
                raise NotImplementedError(
                    f'output_unit {output_unit} is not implemented'
                )

    bandpass_coefficient = (
        np.trapz(bandpass*intensity_derivative_i, freqs)/
        np.trapz(bandpass*intensity_derivative_j, freqs)
    )

    # The coefficient has shape (3,) for conversions to MJy/sr. To allow for 
    # broadcasting the coefficient with the emission we expand its dimensions.
    if bandpass_coefficient.ndim > 0:
        bandpass_coefficient = np.expand_dims(bandpass_coefficient, axis=1)

    return bandpass_coefficient


def _get_interp_parameters(spectral_parameters, n):
    """Returns a dictionary with a regular grid with n points for all spectral
    parameters that vary across the sky.
    """
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





# Intensity derivatives with respect to unit conventions
# ======================================================
def b_rj(freq):
    """The intensity derivative in unit convention K_RJ.

    Args:
    -----
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.   

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        Jy/sr -> K_RJ factor.

    """
    return (2*k_B*freq**2)/(c**2 * u.sr)


def b_cmb(freq, T=T_0):
    """The intensity derivative in unit convention K_CMB.

    Parameters:
    -----------
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.   
    T (astropy.units.quantity.Quantity):
        The CMB emperature. Default is T_0.

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        Bandpass coefficient

    """
    x = (h*freq)/(k_B*T)
    return (
        (2*h*freq**3)/(c**2 * np.expm1(x)) * 
        (np.exp(x)/np.expm1(x))*((h*freq)/(k_B*T**2)) * u.sr**-1
    )


def b_iras(freq, freq_ref):
    """The intensity derivative in the IRAS unit convention (MJy/sr).

    Parameters:
    -----------
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.   

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        Bandpass coefficient

    """
    return (freq_ref/freq)