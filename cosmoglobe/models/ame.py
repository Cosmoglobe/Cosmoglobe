import astropy.units as u
import pathlib
import numpy as np

from .skycomponent import SkyComponent
from .. import data as data_dir

data_path = pathlib.Path(data_dir.__path__[0])

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
    _other_quantities = ('nu_p_map',)

    spdust2_nu, spdust2_amp = np.loadtxt(data_path / 'spdust2_cnm.dat', 
                                         unpack=True)
    spdust2_nu = u.Quantity(spdust2_nu, unit=u.GHz)
    spdust2_amp = u.Quantity(spdust2_amp, unit=(u.Jy/u.sr)).to(
        u.K, equivalencies=u.brightness_temperature(spdust2_nu)
    )

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)
        self._spectral_params = (self.nu_p.si.value,)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu, bandpass=None, output_unit=u.K):
        """
        Extends the get_emission function of the SkyComponent class.

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
            if nu.si.value >= self.spdust2_nu[-1].si.value:
                return np.zeros_like(self.amp)
            if nu.si.value <= self.spdust2_nu[0].si.value:
                raise ValueError(
                                 'Invalid frequency range for AME. '
                                 'AME is not defined below 0.05 GHz.'
                )

        else:
            if (nu.si[-1] >= self.spdust2_nu.si[-1] 
                or nu.si[0] <= self.spdust2_nu.si[0]):
                raise ValueError(
                    'Invalid frequency range for AME when bandpass integrating. '
                )

        return super().get_emission(nu, bandpass, output_unit)


    def _get_freq_scaling(self, nu, nu_p):
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

        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        nu_ref = self.params.nu_ref.si.value
        spdust2_nu = self.spdust2_nu.si.value
        spdust2_amp = self.spdust2_amp.si.value

        scaled_nu = nu * (30e9/nu_p)
        scaled_nu_ref = nu_ref * (30e9/nu_p)

        scaling = np.interp(scaled_nu, spdust2_nu, spdust2_amp)   
        scaling /= np.interp(scaled_nu_ref, spdust2_nu, spdust2_amp)   

        return scaling