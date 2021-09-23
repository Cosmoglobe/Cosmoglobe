from math import pi, log, sqrt
from typing import Any, Protocol, Union, overload
import warnings

from astropy.units import Quantity, Unit
import numpy as np
import healpy as hp

from cosmoglobe.utils.bandpass import (
    get_normalized_bandpass,
    get_bandpass_coefficient,
    get_bandpass_scaling,
)
from cosmoglobe.utils.utils import to_unit, gaussian_beam_2D
from cosmoglobe.sky import DEFAULT_OUTPUT_UNIT, NO_SMOOTHING
from cosmoglobe.sky.basecomponent import (
    SkyComponent,
    DiffuseComponent,
    PointSourceComponent,
)


class SimulationStrategy(Protocol):
    """Abstract base class for simulations strategies.

    Simulation strategies determine how a given type of component is
    simulated in the Cosmoglobe Sky Model."""

    def delta(
        self,
        component: Any,
        freq: Quantity,
        *,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """Method that computes and returns the delta emission.

        Parameters
        ----------
        freq
            Frequency for which to compute the delta emission.

        Returns
        -------
            Emission of the component at a given frequency.
        """

    def bandpass(
        self,
        component: Any,
        freqs: Quantity,
        bandpass: Quantity,
        *,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """Method that computes and returns the bandpass emission.

        Parameters
        ----------
        frequencies
            An array of frequencies for which to compute the bandpass emission
            over.
        bandpass
            An array of bandpass weights matching the frequencies in the
            frequencies array.

        Returns
        -------
            Integrated bandpass emission.
        """


class DiffuseSimulationStrategy:
    """Simulation strategy for diffuse components."""

    def delta(
        self,
        component: DiffuseComponent,
        freq: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

        emission = component.amp * component._get_freq_scaling(
            freq, **component.spectral_parameters
        )

        if output_unit != DEFAULT_OUTPUT_UNIT:
            emission = to_unit(emission, freq, output_unit)

        return emission

    def bandpass(
        self,
        component: DiffuseComponent,
        freqs: Quantity,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

        if bandpass is None:
            warnings.warn("Bandpass is None. Defaulting to top-hat bandpass")
            bandpass = (
                np.ones(freqs_len := len(freqs)) / freqs_len * DEFAULT_OUTPUT_UNIT
            )

        bandpass = get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = get_bandpass_coefficient(bandpass, freqs, output_unit)

        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        emission = component.amp * bandpass_scaling * bandpass_coefficient

        return emission


class PointSourceSimulationStrategy:
    """Simulation strategy for Point Source components."""

    def delta(
        self,
        component: PointSourceComponent,
        freq: Quantity,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """See base class."""

        point_sources = component.amp * component._get_freq_scaling(
            freq, **component.spectral_parameters
        )

        emission = self.pointsources_to_healpix(
            point_sources, component.catalog, nside, fwhm
        )
        emission = to_unit(emission, freq, output_unit)

        return emission

    def bandpass(
        self,
        component: PointSourceComponent,
        freqs: Quantity,
        bandpass: Quantity,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """See base class."""

        if bandpass is None:
            warnings.warn("Bandpass is None. Defaulting to top-hat bandpass")
            bandpass = (
                np.ones(freqs_len := len(freqs)) / freqs_len * DEFAULT_OUTPUT_UNIT
            )

        bandpass = get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = get_bandpass_coefficient(bandpass, freqs, output_unit)
        print(bandpass_coefficient.unit, "UNITUNITUNIT", output_unit)

        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        point_sources = component.amp * bandpass_scaling * bandpass_coefficient
        emission = self.pointsources_to_healpix(
            point_sources, component.catalog, nside, fwhm
        )

        return emission

    def pointsources_to_healpix(
        self, point_sources: Quantity, catalog: np.ndarray, nside: int, fwhm: Quantity
    ) -> Quantity:
        """Maps the point sources to a HEALPIX map.

        For more information, ses `Mitra et al. (2010)
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
        if fwhm.value == NO_SMOOTHING:
            warnings.warn(
                "fwhm not specified. Mapping point sources to pixels "
                "without beam smoothing"
            )
            point_source_pixels = hp.ang2pix(nside, *catalog, lonlat=True)

            print(healpix, "Before")
            for idx, col in enumerate(point_sources):
                healpix[idx, point_source_pixels] = col

            pixel_area = hp.nside2pixarea(nside) * Unit("sr")
            return healpix / pixel_area

        # Applying a truncated beam to each point source
        else:
            sigma = fwhm / (2 * sqrt(2 * log(2)))
            pixel_resolution = hp.nside2resol(nside)
            if fwhm.value < pixel_resolution:
                raise ValueError(
                    "fwhm must be >= pixel resolution to resolve the point sources."
                )

            r_max = N_FWHM * fwhm.value
            pixel_lon, pixel_lat = hp.pix2ang(
                nside, np.arange(hp.nside2npix(nside)), lonlat=True
            )
            beam = gaussian_beam_2D

            print("Smoothing point sources...")
            for idx, (lon, lat) in enumerate(catalog):
                vec = hp.ang2vec(lon, lat, lonlat=True)
                inds = hp.query_disc(nside, vec, r_max)
                r = hp.rotator.angdist(
                    np.array([pixel_lon[inds], pixel_lat[inds]]),
                    np.array([lon, lat]),
                    lonlat=True,
                )
                healpix[inds] += point_sources[idx] * beam(r, sigma.value)

            beam_area = 2 * pi * sigma ** 2
            return healpix / beam_area



# class LineSimulationStrategy:
#     """Simulation strategy for Line components."""

#     def delta(
#         self,
#         component: Line,
#         freq: Quantity,
#         output_unit: Union[str, Unit],
#         **_,
#     ) -> Quantity:
#         """See base class."""
#         ...

#     def bandpass(
#         self,
#         component: Line,
#         freqs: Quantity,
#         bandpass: Quantity,
#         output_unit: Union[str, Unit],
#         **_,
#     ) -> Quantity:
#         """See base class."""
#         ...


def get_simulation_strategy(component: SkyComponent) -> SimulationStrategy:
    """Returns a simulation strategy given a component."""

    if isinstance(component, DiffuseComponent):
        return DiffuseSimulationStrategy()
    # elif isinstance(component, Line):
    #     return LineSimulationStrategy()
    elif isinstance(component, PointSourceComponent):
        return PointSourceSimulationStrategy()
    else:
        raise NotImplementedError
