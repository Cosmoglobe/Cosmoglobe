import astropy.units as u
import astropy.constants as const
import healpy as hp
import numpy as np
import os
import time 

from .. import data as data_dir

h = const.h
c = const.c
k_B = const.k_B
T_0 = 2.7255*u.K

data_path = os.path.dirname(data_dir.__file__) + '/'


# def trapezoidal_step(f, )


@u.quantity_input(input_map=u.K, nu=u.Hz)
def KRJ_to_KCMB(input_map, nu):
    """
    Converts input map from units of K_RJ to K_CMB.
    Parameters
    ----------
    input_map : astropy.units.quantity.Quantity
        Healpix map in units of K_RJ.
    Returns
    -------
    astropy.units.quantity.Quantity
        Output map in units of K_CMB.
    """
    x = ((h*nu) / (k_B*T_0)).si.value
    scaling_factor = (np.expm1(x)**2) / (x**2 * np.exp(x))

    return input_map*scaling_factor


@u.quantity_input(input_map=u.K, nu=u.Hz)
def KCMB_to_KRJ(input_map, nu):
    """
    Converts input map from units of K_CMB to K_RJ.
    Parameters
    ----------
    input_map : astropy.units.quantity.Quantity
        Healpix map in units of K_RJ.
    Returns
    -------
    astropy.units.quantity.Quantity
        Output map in units of K_CMB.
    """
    x = ((h*nu) / (k_B*T_0)).si.value
    scaling_factor = (np.expm1(x)**2) / (x**2 * np.exp(x))

    return input_map/scaling_factor


def create_70GHz_mask(sky_frac):
    """
    Creates a mask from the 70GHz BP7 frequency map at nside=256 for a 
    threshold corresponding to a given sky fraction (in %). The 70GGz
    map is chosen due to it containing a large portion of all low-frequency
    sky components.

    Parameters
    ----------
    sky_frac : float
        Sky fraction in percentage.

    Returns
    -------
    mask : numpy.ndarray
        Mask covering the sky for a given sky fraction.

    """
    template = hp.read_map(data_path + 'BP7_70GHz_nocmb_n0256.fits', 
                           dtype=np.float64, verbose=False)
    template = hp.ma(template)

    # Masking based on sky fraction is not trivial. Here we manually compute
    # the sky fraction by masking all pixels with amplitudes larger than a 
    # given percentage of the maximum map amplitude. The pixels masks then 
    # correspond to a sky fraction. We tabulate the amplitude percentage and
    # sky fraction for a range, and interpolate from this table.
    amp_percentages = np.flip(np.arange(1,101))
    fracs = []
    mask = np.zeros(len(template), dtype=np.bool)

    for i in range(len(amp_percentages)):
        mask = np.zeros(len(template), dtype=np.bool)
        masked_template = np.abs(hp.ma(template))
        mask[np.where(np.log(masked_template) > 
            (amp_percentages[i]/100)*np.nanmax(np.log(masked_template)))] = 1
        masked_template.mask = mask

        frac = ((len(masked_template)-masked_template.count())
                /len(masked_template))*100
        fracs.append(frac)

    amp_percentage = np.interp(100-sky_frac, fracs, amp_percentages)

    mask = np.zeros(len(template), dtype=np.bool)
    masked_template = np.abs(hp.ma(template))
    mask[np.where(np.log(masked_template) > 
        (amp_percentage/100)*np.nanmax(np.log(masked_template)))] = 1

    return mask 


def timer(function):
    """
    Timer decorator for development purposes.

    """
    def wrapper(*args, **kwargs):
        start = time.time()
        return_value = function(*args, **kwargs)
        total_time = time.time() - start
        print(f'{function.__name__} took: {total_time} s')
        return return_value

    return wrapper