import astropy.units as u
import astropy.constants as const
import healpy as hp
import numpy as np
import pathlib
import time 
import inspect

from .. import data as data_dir

h = const.h
c = const.c
k_B = const.k_B
T_0 = 2.7255*u.K

data_path = pathlib.Path(data_dir.__path__[0])


@u.quantity_input(nus=u.Hz, bandpass=(u.Jy/u.sr, u.K))
def get_normalized_bandpass(nus, bandpass):
    """
    Converts the bandpass profile to 'K_RJ' units and normalizes it unity 
    under the trapezoidal integration.
    Parameters
    ----------
    nus : astropy.units.quantity.Quantity
        Array or list of frequencies for the bandpass profile.
    bandpass : astropy.units.quantity.Quantity
        Bandpass profile array or list. Must be in units of Jy/sr or K_RJ.
    Returns
    -------
    bandpass : numpy.ndarray
        Normalized bandpass profile to unit integral in units of K_RJ.
        
    """
    if np.shape(nus) != np.shape(bandpass):
        raise ValueError(
            'Frequency and bandpass arrays must have the same shape'
        )

    bandpass = bandpass.to_value(u.K, equivalencies=u.brightness_temperature(nus))
    bandpass /= np.trapz(bandpass, nus.si.value)

    return bandpass * u.K


def get_unit_conversion(nus, bandpass, output_unit):
    """
    Returns the unit conversion factor for a bandpass integration given an 
    output and input unit.
    Parameters
    ----------
    nus : astropy.units.quantity.Quantity
        Frequency array or list for the bandpass profile.
    bandpass : astropy.units.quantity.Quantity
        Bandpass profile array or list in units K_RJ.
    output_unit : astropy.units.core.IrreducibleUnit or 
                  astropy.units.core.CompositeUnit
        Output unit of bandpass integrated map.
    Returns
    -------
    weights : numpy.ndarray
        Normalized bandpass profile to unit integral.
        
    """
    bandpass_input = bandpass.value
    bandpass_output = (bandpass.value * output_unit).to_value(u.K, 
        equivalencies=u.brightness_temperature(nus)
    )

    factor = np.trapz(bandpass_input, nus.si.value)
    factor /= np.trapz(bandpass_output, nus.si.value)

    return factor


def get_interp_ranges(spectral_params, n):
    """
    Returns interpolation information required to determine and compute the
    n-dimensional interpolation in the mixing matrix.

    Parameters
    ----------
    spectral_params : list, tuple, numpy.ndarray
        Spectral parameters for a model.
    n : int
        Number of interpolation points.

    Returns
    -------
    interpol_ranges : list
        List linearly spaced interpolation arrays for all spectral parameters 
        varying over the sky.
    consts : list
        List of the scalar value of all spectral parameters constant over the 
        sky.
    interp_dim : int
        Interpolation dimension. Is decided by the number of items in the
        interpol_ranges list.
    interp_ind : int
        If interp_dim == 1, but len(spectral_params) > 1, interp_ind is the 
        index of spectral parameter in spectral_params that we want to 
        interpolate over. If interp_dim != 1, interp_ind = None.
    
    """
    interp_dim = 0
    interp_ind = None
    interpol_ranges, consts = [], []
    if not isinstance(spectral_params, (tuple, list)):
        raise TypeError(
            'spectral_params argument must be of type tuple or list'
        )
        
    if len(spectral_params) == 1:
        unique_values = np.unique(spectral_params)
        if len(unique_values) > 1:
            interp_dim += 1
            interp_ind = 0

            min_value = np.amin(spectral_params)
            max_value = np.amax(spectral_params)
            interpol_ranges.append(np.linspace(min_value, max_value, n))

        else:
            consts.append(unique_values[0])
    
    else:
        for i, param in enumerate(spectral_params):
            unique_values = np.unique(param)
            if len(unique_values) > 1:
                interp_dim += 1
                if interp_ind is None:
                    interp_ind = i

                min_value = np.amin(param)
                max_value = np.amax(param)
                interpol_range = np.linspace(min_value, max_value, n)
                interpol_ranges.append(interpol_range)
            
            else:
                consts.append(unique_values[0])

    return interpol_ranges, consts, interp_dim, interp_ind


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
    template = hp.read_map(data_path / 'BP7_70GHz_nocmb_n0256.fits', 
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


def to_IQU(array):
    """
    Takes in a array of size (,nside) and reutuns the same array with new 
    shape (3, nside), where the original array resides in the first dimension, 
    while dimensions two and three are empty.

    """
    
    if array.ndim > 1:
        new_array = np.zeros((3, len(array[0])))
        new_array[0] = array[0]
    else:
        new_array = np.zeros((3, len(array)))
        new_array[0] = array
    return new_array


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


def nside_isvalid(function):
    """
    Decorator that validates the nside input to a function. Function must have
    a keyword named 'nside'.

    """
    def wrapper(*args, **kwargs):
        if kwargs:
            nside = kwargs['nside']
        else:
            arg_names = inspect.getargspec(function).args
            try:
                nside_ind = arg_names.index('nside')
            except KeyError:
                raise KeyError(
                f'Method {function.__name__} has no nside argument'
            )
            nside = args[nside_ind]

        if nside is not None and not hp.isnsideok(nside, nest=True):
            raise ValueError(f'nside: {nside} is not valid.')

        return_value = function(*args, **kwargs)

        return return_value
    return wrapper