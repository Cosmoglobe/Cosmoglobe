from .skycomponent import SkyComponent

import numpy as np
import astropy.units as u
import astropy.constants as const

# Initializing common astrophy units
h = const.h
c = const.c
k_B = const.k_B


class Dust(SkyComponent):
    """
    Parent class for all Dust models.
    """
    comp_label = 'dust'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs



class ModifiedBlackbody(Dust):
    """
    Model for modifed blackbody (mdd) emission for thermal dust.

    """    
    model_label = 'MBB'

    def __init__(self, data):
        super().__init__(data)


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
        emission = self.compute_emission(nu, self.params['nu_ref'], 
                                         self.beta, self.T)

        return emission


    @u.quantity_input(nu=u.Hz, nu_ref=u.Hz, T=u.K)
    def compute_emission(self, nu, nu_ref, beta, T):
        """
        Computes the simulated modified blackbody emission of dust at a given
        frequency .

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the modified blackbody radiation. 
        nu_ref : 'astropy.units.quantity.Quantity'
            Reference frequency at which the Commander alm outputs are 
            produced.        
        beta : 'numpy.ndarray'
            Estimated dust beta map from Commander.
        T : 'astropy.units.quantity.Quantity'
            Temperature the modified blackbody. 
            
        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Modeled modified blackbody emission at a given frequency.

        """
        emission = self.amp.copy()
        for i in range(len(emission)):
            emission[i] *= (nu/nu_ref[i])**beta[i]
            emission[i] *= blackbody_ratio(nu, nu_ref[i], T[i])

        return emission


@u.quantity_input(nu=u.Hz, nu_ref=u.Hz, T=u.K)
def blackbody_ratio(nu, nu_ref, T):
    """
    Returns the ratio between the emission emitted by a blackbody at two
    different frequencies.

    Parameters
    ----------
    nu : 'astropy.units.quantity.Quantity'
        New frequency at which to evaluate the blackbody radiation.
    nu_ref : 'astropy.units.quantity.Quantity'
        Reference frequency at which the Commander alm outputs are 
        produced.        
    T : 'astropy.units.quantity.Quantity'
        Temperature of the blackbody.

    Returns
    -------
    ratio : float
        Ratio between the blackbody emission at two different frequencies.

    """
    ratio = blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)

    return ratio

@u.quantity_input(nu=u.Hz, T=u.K)
def blackbody_emission(nu, T):
    """
    Returns blackbody radiation at a given frequency and temperature
    in units of uK_RJ.

    Parameters
    ----------
    nu : 'astropy.units.quantity.Quantity'
        Frequency at which to evaluate the blackbody radiation.
    T : 'astropy.units.quantity.Quantity'
        Temperature of the blackbody. 

    Returns
    -------
    'astropy.units.quantity.Quantity'
        Blackbody emission in units of uK_RJ.

    """
    emission = 1 / np.expm1(h*nu/(k_B*T))
    emission *= 2*h*nu**3 / c**2
    emission /= u.sr    # adds astropy steradians unit

    return emission.to(u.uK, equivalencies=u.brightness_temperature(nu))