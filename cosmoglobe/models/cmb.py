import astropy.units as u
import astropy.constants as const
import numpy as np

from .skycomponent import SkyComponent

h = const.h
c = const.c
k_B = const.k_B
T_0 = 2.7255*u.K


class CMB(SkyComponent):
    """
    Parent class for all CMB models.
    
    """
    comp_label = 'cmb'
    _multipoles = (0, 1)
    
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs

        if kwargs.get('remove_monopole', False):
            self.amp -= self.monopole

        if kwargs.get('remove_dipole', False):
            self.amp -= self.dipole



class BlackBody(CMB):
    """
    Model for BlackBody CMB emission.
    
    """
    model_label = 'cmb'

    def __init__(self, data, nside=None, fwhm=None,
                 remove_monopole=True, remove_dipole=True):
        super().__init__(data, nside=nside, fwhm=fwhm, 
                         remove_monopole=remove_monopole, 
                         remove_dipole=remove_dipole)
        self._spectral_params = ()


    def _get_freq_scaling(self, nu):
        """
        Computes the frequency scaling from the reference frequency nu_ref to 
        an arbitrary frequency nu, which depends on the spectral parameter
        beta.

        Parameters
        ----------
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency.        
        beta : numpy.ndarray
            Synch beta map.
            
        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        x = ((h*nu) / (k_B*T_0)).si.value
        scaling_factor = (x**2 * np.exp(x)) / (np.expm1(x)**2)
        return scaling_factor