from .skycomponent import SkyComponent
from .. import data as data_dir

import astropy.units as u
import importlib.resources as pkg_resources
import numpy as np


class AME(SkyComponent):
    """
    Parent class for all AME models.
    """
    comp_label = 'ame'

    def __init__(self, data):
        super().__init__(data)



class SpinningDust2(AME):
    """
    Model for spinning dust emission.

    """    
    model_label = 'spindust2'

    def __init__(self, data):
        super().__init__(data)

        # Reading in spdust2 template to interpolate for arbitrary frequency
        spdust2_spectrum = pkg_resources.read_text(data_dir, 'spdust2_cnm.dat')
        self.spdust2_nu = spdust2_spectrum[0]
        self.spdust2_amp = spdust2_spectrum[1]
        self.nu_p = data.get_item(self.comp_label, 'nu_p_map')*u.GHz


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the model. 

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Model emission at given frequency in units of K_RJ.

        """
        emission = self.compute_emission(nu, self.params['nu_ref'], self.nu_p, 
                                         self.spdust2_amp, self.spdust2_nu)

        return emission


    @u.quantity_input(nu=u.Hz, nu_ref=u.Hz, nu_p=u.Hz)
    def compute_emission(self, nu, nu_ref, nu_p, spdust2_amp, spdust2_nu):
        """
        Computes the simulated spinning dust emission at a given frequency.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency (GHz) at which to evaluate the modified blackbody radiation. 
        nu_ref : 'astropy.units.quantity.Quantity'
            Reference frequency (GHz) at which the Commander alm outputs are 
            produced.
        nu_p : 'astropy.units.quantity.Quantity'
            Peak frequency (GHz) parameter that shifts the specrum in log nu -
            log s space.
        spdust2_amp : 'numpy.ndarray'
            SpDust2 code template amplitude. See Ali-Haimoud et al. 2009 for more 
            information.        
        spdust2_nu : 'numpy.ndarray'
            SpDust2 code template frequency. See Ali-Haimoud et al. 2009 for more 
            information.

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Modeled modified blackbody emission at a given frequency.

        """
        emission = self.amp.copy()
        scaled_nu = (nu * (30*u.GHz / nu_p)).value
        for i in range(len(emission)):
            scaled_ref_nu = (nu_ref[i] * (30*u.GHz / nu_p)).value
            emission[i] *= np.interp(scaled_nu, spdust2_amp, spdust2_nu)   
            emission[i] /= np.interp(scaled_ref_nu[i], spdust2_amp, spdust2_nu)    
            emission[i] *= (nu_ref[i]/nu)**2

        return emission

