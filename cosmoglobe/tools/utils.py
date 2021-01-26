from .. import data as data_dir
import healpy as hp
import os
import numpy as np

data_path = os.path.dirname(data_dir.__file__) + '/'


def create_mask(sky_frac):
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