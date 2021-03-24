from .constants import h, c, k_B
import numpy as np

def blackbody_emission(nu, T):
    """Returns the emission emitted by a blackbody with with temperature T at 
    a frequency nu in SI units.

    Args:
    -----
    nu : int, float, numpy.ndarray
        Frequency at which to evaluate the blackbody radiation.
    T : int, float, numpy.ndarray
        Temperature of the blackbody. 

    Returns:
    --------
    int, float, numpy.ndarray
        Blackbody emission in units of Jy/sr

    """
    return ((2*h*nu**3)/c**2) / np.expm1((h*nu) / (k_B*T))