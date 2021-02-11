import astropy.constants as const
import astropy.units as u
import numpy as np

from .skycomponent import SkyComponent

h = const.h
k_B = const.k_B


class FreeFree(SkyComponent):
    """
    Parent class for all Free-Free models.
    
    """
    comp_label = 'ff'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs


class LinearOpticallyThin(FreeFree):
    """
    Linearized model only valid in the optically thin case (tau << 1).
    
    """
    model_label= 'freefree'
    _other_quantities = ('Te_map',)

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=u.K):
        """
        Returns the model emission at an arbitrary frequency nu in units 
        of K_RJ.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the model.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile in units of (k_RJ, Jy/sr) corresponding to 
            frequency array nu. If None, a delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.quantity.Quantity or str
            Desired unit for the output map. Must be a valid astropy.unit or 
            one of the two following strings ('K_CMB', 'K_RJ').
            Default : None

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        if bandpass is None:
            scaling = self._get_freq_scaling(nu.si.value, self.Te_map.value)
            emission = self.amp*scaling

        else:
            bandpass = self._get_normalized_bandpass(nu, bandpass)
            U = self._get_unit_conversion(nu, bandpass, output_unit)
            M = self._get_mixing(bandpass=bandpass.value,
                                 nus=nu.si.value, 
                                 spectral_params=(self.Te_map.value,))

            emission = self.amp*M*U

        return emission


    def _get_freq_scaling(self, nu, T_e):
        """
        Computes the frequency scaling from the reference frequency nu_ref to 
        an arbitrary frequency nu, which depends on the spectral parameters
        T_e.

        Parameters
        ----------
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency.        
        T_e : numpy.ndarray
            Electron temperature map.

        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        # scaling = np.exp(-h * ((nu-nu_ref) / (k_B*T_e)))
        nu_ref = self.params['nu_ref'].si.value

        # Commander outputs generally in type float32 which results in overflow 
        # in the calculation of the gaunt factor
        T_e = T_e.astype(np.float64)

        scaling = gaunt_factor(nu, T_e) / gaunt_factor(nu_ref, T_e)
        scaling *= (nu_ref/nu)**2

        return scaling


def gaunt_factor(nu, T_e):
    """
    Returns the gaunt factor for a given frequency and electron temperature.

    Parameters
    ----------
    nu : int, float, numpy.ndarray
        Frequency at which to evaluate the Gaunt factor.   
    T_e : numpy.ndarray
        Electron temperature at which to evaluate the Gaunt factor.

    Returns
    -------
    numpy.ndarray
        Gaunt Factor.

    """
    return np.log(np.exp(5.96 - (np.sqrt(3)/np.pi) * np.log(nu
                  * (T_e*1e-4)**-1.5)) + np.e)
