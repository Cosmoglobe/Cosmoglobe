import astropy.units as u
from numba import njit
import numpy as np
import os


from .. import data as data_dir
from .skycomponent import SkyComponent

data_path = os.path.dirname(data_dir.__file__) + '/'


class AME(SkyComponent):
    """
    Parent class for all AME models.
    
    """
    comp_label = 'ame'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs


class SpinningDust2(AME):
    """
    Model for spinning dust emission.
    
    """    
    model_label = 'spindust2'
    other_quantities = ('nu_p_map',)

    # Reading in spdust2 template to interpolate for arbitrary frequency
    spdust2_nu, spdust2_amp = np.loadtxt(data_path + 'spdust2_cnm.dat', 
                                         unpack=True)

    spdust2_nu = spdust2_nu * u.GHz
    spdust2_amp = u.Quantity(spdust2_amp, unit=(u.Jy/u.sr)).to(
        u.K, 
        equivalencies=u.brightness_temperature(spdust2_nu)
    )

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission at an arbitrary frequency nu in units 
        of K_RJ.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            Frequencies at which to evaluate the model. 

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        if nu.si.value >= self.spdust2_nu[-1].si.value:
            return u.Quantity(np.zeros_like(self.amp), unit=self.amp.unit)
        
        if nu.si.value <= self.spdust2_nu[0].si.value:
            raise ValueError(
                             'Invalid frequency range for AME. '
                             'AME is not defined below 0.05 GHz.'
            )

        scaling = self._get_freq_scaling(nu.si.value, 
                                         self.params['nu_ref'].si.value, 
                                         self.nu_p_map.si.value, 
                                         self.spdust2_amp.value, 
                                         self.spdust2_nu.si.value)
        emission = self.amp*scaling

        return emission


    @staticmethod
    @njit
    def _get_freq_scaling(nu, nu_ref, nu_p, spdust2_amp, spdust2_nu):
        """
        Computes the frequency scaling from the reference frequency nu_ref to 
        an arbitrary frequency nu, given by the peak frequency nu_p abd
        SED template from theSpDust2 code (see Ali-Haimoud et al. 2009).

        Parameters
        ----------
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency.    
        nu_p : numpy.ndarray
            Peak position parameter.    
        spdust2_amp : numpy.ndarray
            SpDust2 code template amplitude.
        spdust2_nu : numpy.ndarray
            SpDust2 code template frequency.

        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        scaled_nu = nu * (30e9/nu_p)
        scaled_nu_ref = nu_ref * (30e9/nu_p)

        scaling = np.interp(scaled_nu, spdust2_nu, spdust2_amp)   
        scaling /= np.interp(scaled_nu_ref, spdust2_nu, spdust2_amp)   

        return scaling

