from math import pi, log, sqrt
import warnings

from astropy.units import Quantity, Unit, quantity_input
import numpy as np
import healpy as hp

from cosmoglobe.utils.utils import gaussian_beam_2D
from cosmoglobe.sky._constants import DEFAULT_BEAM_FWHM

TEST_BEAM_BL = "/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/wmap_beam.txt"



@quantity_input(fwhm=Unit("rad"))
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


def pointsources_to_healpix(
    point_sources: Quantity,
    catalog: np.ndarray,
    nside: int,
    fwhm: Quantity,
) -> Quantity:
    """Maps the point sources to a HEALPIX map.

    For more information on this calculation, please see `Mitra et al. (2010)
    <https://arxiv.org/pdf/1005.1929.pdf>`_.
    """

    N_FWHM = 2  # FWHM cutoff for the truncated beam

    healpix = Quantity(
        np.zeros((point_sources.shape[0], hp.nside2npix(nside))),
        unit=point_sources.unit,
    )
    # Getting the longitude and latitude for each pixel on the healpix map

    fwhm = fwhm.to("rad")
    # Directly map to pixels without any smoothing
    if fwhm.value == DEFAULT_BEAM_FWHM:
        warnings.warn(
            "fwhm not specified. Mapping point sources to pixels "
            "without beam smoothing"
        )
        point_source_pixels = hp.ang2pix(nside, *catalog, lonlat=True)
        for IQU, emission in enumerate(point_sources):
            healpix[IQU, point_source_pixels] = emission

        pixel_area = hp.nside2pixarea(nside) * Unit("sr")
        return healpix / pixel_area

    # Applying a truncated beam to each point source
    else:
        sigma = get_sigma(fwhm)
        pixel_resolution = hp.nside2resol(nside)
        if fwhm.value < pixel_resolution:
            raise ValueError(
                "fwhm must be >= pixel resolution to resolve the point sources."
            )

        r_max = N_FWHM * fwhm.value
        pixel_lon, pixel_lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True
        )

        print("Smoothing point sources...")
        lons, lats = catalog
        for idx, (lon, lat) in enumerate(zip(lons, lats)):
            vec = hp.ang2vec(lon, lat, lonlat=True)
            pixels = hp.query_disc(nside, vec, r_max)
            angular_distance = hp.rotator.angdist(
                [pixel_lon[pixels], pixel_lat[pixels]], [lon, lat], lonlat=True
            )
            beam = gaussian_beam_2D(angular_distance, sigma.value)
            for IQU, emission in enumerate(point_sources):
                healpix[IQU, pixels] += emission[idx] * beam

        beam_area = 2 * pi * sigma ** 2
        return healpix / beam_area


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    b_l = np.loadtxt(TEST_BEAM_BL, skiprows=10)
    beam = hp.bl2beam(b_l[:, 1], b_l[:, 0])
    plt.plot(b_l[:, 0 ], beam)
    plt.show()
