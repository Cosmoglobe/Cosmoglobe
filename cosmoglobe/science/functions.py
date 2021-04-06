from .constants import h, c, k_B, T_0
import numpy as np
import astropy.units as u

def blackbody_emission(freq, T):
    """Returns the emission emitted by a blackbody with with temperature T at 
    a frequency freq in SI units.

    Args:
    -----
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.
    T (astropy.units.quantity.Quantity):
        Temperature of the blackbody in units of K. 

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        Blackbody emission in units in SI units.

    """
    return ((2*h*freq**3)/c**2) / np.expm1((h*freq) / (k_B*T))


def gaunt_factor(freq, T_e):
    """Returns the gaunt factor at a given frequency and electron temperature
    in SI units.

    Args:
    -----
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.   
    T_e (astropy.units.quantity.Quantity):
        Electron temperature in units of K.

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        Gaunt Factor.

    """
    try:
        T_e = T_e.astype(np.float64)
    except AttributeError:
        pass

    try:
        freq = freq.si.value    
        T_e = T_e.si.value
    except AttributeError:
        pass    
        
    gaunt_factor = np.log(np.exp(5.96 - (np.sqrt(3)/np.pi) * np.log(freq
                  * (T_e*1e-4)**-1.5)) + np.e)

    return u.Quantity(gaunt_factor)



def cmb_to_brightness(freq):
    """Returns the conversion factor between CMB and brightness temperatures 
    (K_CMB and K_RJ).

    TODO: rename this to a better fitting name?

    Args:
    -----
    freq (astropy.units.quantity.Quantity):
        Frequency in units of Hertz.   

    Returns:
    --------
    (astropy.units.quantity.Quantity):
        K_CMB -> K_RJ factor.

    """  
    x = ((h*freq) / (k_B*T_0))
    return (x**2 * np.exp(x)) / (np.expm1(x)**2)