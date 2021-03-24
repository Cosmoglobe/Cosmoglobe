from ..sky.map import to_IQU
from ..science.functions import blackbody_emission


import numpy as np
import healpy as hp
import astropy.units as u
import astropy.constants as const

h = const.h.value
c = const.c.value
k_B = const.k_B.value

class Component:
    """Base class for all sky components.

    Any sky component you make should subclass this class.

    Args:
    -----
    amp : astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        Amplitude map for I, Q and U stokes parameters.
    nu_ref : tuple, list, np.ndarray
        Reference frequencies for the amplitude map. Each array element must 
        be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    spectrals: dict
        Spectral maps required to compute the frequency scaling factor. Maps 
        must be of types astropy.units.quantity.Quantity or cosmoglobe.IQUMap.

    """
    def __init__(self, amp, nu_ref, **spectrals):
        self.amp = to_IQU(amp)
        self.nu_ref = nu_ref

        self.spectrals = {}
        if spectrals is not None:
            for key, value in spectrals.items():
                self.spectrals[key] = to_IQU(value)
        

    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=None):
        """
        Returns the full sky component emission at an arbitrary frequency nu.

        Args:
        -----
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the 
            component emission.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile in units of K_RJ or Jy/sr corresponding to the
            frequency array, nu. If None, a delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.quantity.Quantity or str
            Desired unit for the output map. Must be a valid astropy.unit or 
            one of the two following strings 'K_CMB', 'K_RJ'.
            Default : None

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        if bandpass is None:
            scaling = self.get_freq_scaling(nu.si.value, **self.spectrals)
            emission = self.amp*scaling

        # else:
        #     bandpass = utils.get_normalized_bandpass(nu, bandpass)
        #     U = utils.get_unit_conversion(nu, bandpass, output_unit)
        #     M = self._get_mixing(bandpass=bandpass.value,
        #                          nus=nu.si.value, 
        #                          spectral_params=self._spectral_params)
        #     emission = self.amp*M*U

        return emission


    def __repr__(self):
        main_repr = f'{self.__class__.__name__}'
        main_repr += '('
        main_repr += 'amp, '
        main_repr += 'nu_ref, '
        for key, value in self.spectrals.items():
            main_repr += f'{key}, '
        main_repr = main_repr[:-2]
        main_repr += ')'

        return main_repr




class PowerLaw(Component):
    """PowerLaw component class.

    Args:
    -----
    amp : astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        PowerLaw IQU amplitude map.
    nu_ref : tuple, list, np.ndarray
        Reference frequencies for the synch amplitude map. Each array 
        element must be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    beta: astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        PowerLaw IQU beta map.

    """
    def __init__(self, amp, nu_ref, beta):
        super().__init__(amp, nu_ref, beta=beta)


    def get_freq_scaling(self, nu, beta):
        """Computes the frequency scaling from the reference frequency nu_ref 
        to an arbitrary frequency nu, which depends on the spectral parameter
        beta.

        Args:
        -----
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. Must be in si values.      
        beta : numpy.ndarray
            Synch beta map. Must be dimensionless.
            
        Returns:
        --------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        nu_ref = np.expand_dims(self.nu_ref.si.value, axis=1)
        scaling = (nu/nu_ref)**beta

        return scaling


    
class ModifiedBlackBody(Component):
    """Modified BlackBody (MBB) component class.

    Args:
    -----
    amp : astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        MBB IQU amplitude map.
    nu_ref : tuple, list, np.ndarray
        Reference frequencies for the synch amplitude map. Each array 
        element must be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    beta: astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        MBB IQU beta map.

    """

    def __init__(self, amp, nu_ref, beta, T):
        super().__init__(amp, nu_ref, beta=beta, T=T)

    def get_freq_scaling(self, nu, beta, T):
        """Computes the frequency scaling from the reference frequency nu_ref 
        to an arbitrary frequency nu, which depends on the spectral parameters
        beta and T.

        Args:
        -----
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        beta : numpy.ndarray
            MBB beta map. Must be dimensionless.
        T : numpy.ndarray
            MBB temperature map with unit K.

        Returns:
        --------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        nu_ref = np.expand_dims(self.params.nu_ref.si.value, axis=1)

        scaling = (nu/nu_ref)**(beta-2)
        scaling *= blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)

        return scaling