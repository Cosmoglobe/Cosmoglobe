from .skycomponent import SkyComponent

import numpy as np
import astropy.units as u
import astropy.constants as const

# Initializing common astrophy units
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
    Linearized model strictly only valid in the optically thin case (tau << 1).

    """    
    model_has_polarization = False
    model_label= 'freefree'

    def __init__(self, data):
        super().__init__(data)

        self.T_e = data.get_item(self.comp_label, 'Te_map')* u.K


    @u.quantity_input(nu=u.GHz)
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
        emission = self.compute_emission(nu, self.params['nu_ref'], self.T_e)

        return emission

    @u.quantity_input(nu=u.GHz, nu_ref=u.GHz, T_e=u.K)
    def compute_emission(self, nu, nu_ref, T_e):
        """
        Computes the power law emission at a given frequency for the 
        given Commander data.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the modified blackbody radiation. 
        nu_ref : 'astropy.units.quantity.Quantity'
            Reference frequency at which the Commander alm outputs are 
            estimated.        
        T_e : 'astropy.units.quantity.Quantity'

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Modeled modified blackbody emission at a given frequency.

        """
        emission = self.amp.copy()
        for i in range(len(emission)):
            g_eff = gaunt_factor(nu, T_e[i])
            g_eff_ref = gaunt_factor(nu_ref[i], T_e[i])
            emission[i] *= g_eff/g_eff_ref
            emission[i] *= np.exp(-h * ((nu-nu_ref[i])/(k_B*T_e[i])))
            emission[i] *= (nu/nu_ref[i])**-2

        return emission


def gaunt_factor(nu, T_e):
    """
    Computes the caunt factor for a given frequency and electron temperature.

    Parameters
    ----------
    nu : 'astropy.units.quantity.Quantity'
        Frequency at which to evaluate the Gaunt factor.   
    T_e : 'astropy.units.quantity.Quantity'
        Electron temperature at which to evaluate the Gaunt factor.

    Returns
    -------
    double, 'numpy.ndarray'
        Gaunt Factor.

    """
    return np.log(np.exp(5.96-(np.sqrt(3)/np.pi) * np.log(nu.value*
                  (T_e.value*1e-4)**-1.5)) + np.e )
