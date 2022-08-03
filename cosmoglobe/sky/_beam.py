import warnings
from math import log, pi, sqrt

import healpy as hp
import numpy as np
from astropy.units import Quantity, Unit, quantity_input

from cosmoglobe.sky._constants import DEFAULT_BEAM_FWHM


@quantity_input(fwhm=("rad", "arcmin", "deg"))
def get_sigma(fwhm: Quantity) -> Quantity:
    """Returns the standard deviation given a fwhm.

    Parameters
    ----------
    fwhm
        FWHM of the beam.

    Returns
    -------
        Standard deviation of the beam.
    """

    return fwhm / (2 * sqrt(2 * log(2)))


def gaussian_beam_2D(r: np.ndarray, sigma: float) -> np.ndarray:
    """Returns the Gaussian beam in 2D in polar coordinates.

    Parameters
    ----------
    r
        Angular distance.
    sigma
        The sigma of the Gaussian (beam radius).

    Returns
    -------
        Gaussian beam.
    """

    return r * np.exp(-(r ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def pointsources_to_healpix(
    point_sources: Quantity,
    catalog: Quantity,
    nside: int,
    fwhm: Quantity,
) -> Quantity:
    """Maps the point sources to a HEALPIX map.

    For more information on this calculation, please see `Mitra et al. (2010)
    <https://arxiv.org/pdf/1005.1929.pdf>`_.
    """

    N_FWHM = 2  # FWHM cutoff for the truncated beam

    healpix_map = Quantity(
        np.zeros((point_sources.shape[0], hp.nside2npix(nside))),
        unit=point_sources.unit,
    )

    # Getting the longitude and latitude for each pixel on the healpix map

    fwhm = fwhm.to("rad")
    catalog = catalog.to("deg").value

    # Directly map to pixels without any smoothing
    if fwhm == DEFAULT_BEAM_FWHM:
        warnings.warn(
            "fwhm not specified. Mapping point sources to pixels "
            "without beam smoothing"
        )
        pixels = hp.ang2pix(nside, *catalog, lonlat=True)
        for IQU, emission in enumerate(point_sources):
            healpix_map[IQU, pixels] = emission

        pixel_area = hp.nside2pixarea(nside) * Unit("sr")
        return healpix_map / pixel_area

    # Applying a truncated Gaussian beam to each point source
    else:
        sigma = get_sigma(fwhm)
        if fwhm.value < hp.nside2resol(nside):
            raise ValueError(
                "fwhm must be >= pixel resolution to resolve the point sources."
            )

        r_max = N_FWHM * fwhm.value
        pixel_lon, pixel_lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True
        )

        lons, lats = catalog
        for idx, (lon, lat) in enumerate(zip(lons, lats)):
            vec = hp.ang2vec(lon, lat, lonlat=True)
            pixels = hp.query_disc(nside, vec, r_max)
            angular_distance = hp.rotator.angdist(
                [pixel_lon[pixels], pixel_lat[pixels]], [lon, lat], lonlat=True
            )
            beam = gaussian_beam_2D(angular_distance, sigma.value)
            for IQU, emission in enumerate(point_sources):
                healpix_map[IQU, pixels] += emission[idx] * beam

        beam_area = 2 * pi * sigma ** 2

        return healpix_map / beam_area
