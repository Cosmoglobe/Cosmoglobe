from typing import Protocol, Union
import warnings

from astropy.units import Quantity, Unit
import numpy as np

from cosmoglobe.utils.bandpass import (
    get_bandpass_scaling,
)
from cosmoglobe.utils.utils import to_unit
from cosmoglobe.sky.base_components import (
    SkyComponent,
    DiffuseComponent,
    PointSourceComponent,
    LineComponent,
)
from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky._bandpass import get_normalized_bandpass, get_bandpass_coefficient
from cosmoglobe.sky._beam import pointsources_to_healpix

SkyComponent = Union[DiffuseComponent, PointSourceComponent, LineComponent]


class Simulation(Protocol):
    """Protocol defining how a component behaves when used in simulations."""

    def delta(
        self,
        component: SkyComponent,
        freq: Quantity,
        *,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """Computes and returns the emission for a delta frequency.

        Parameters
        ----------
        component
            A SkyComponent.
        freq
            Frequency for which to compute the delta emission.
        nside
            nside of the HEALPIX map.
        fwhm
            The full width half max parameter of the Gaussian.
        output_unit
            The requested output units of the simulated emission.

        Returns
        -------
            Simulated delta frequency emission.
        """

    def bandpass(
        self,
        component: SkyComponent,
        freqs: Quantity,
        bandpass: Quantity,
        *,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """Computes and returns the emission for a bandpass profile.

        Parameters
        ----------
        freqs
            An array of frequencies for which to compute the bandpass emission
            over.
        bandpass
            An array of bandpass weights corresponding the frequencies in the
            frequencies array.
        nside
            nside of the HEALPIX map.
        fwhm
            The full width half max parameter of the Gaussian.
        output_unit
            The requested output units of the simulated emission.

        Returns
        -------
            Integrated bandpass emission.
        """


class DiffuseSimulation:
    """Simulation protocol for diffuse components."""

    def delta(
        self,
        component: DiffuseComponent,
        freq: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

        emission = component.amp * component.get_freq_scaling(
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

        bandpass = get_normalized_bandpass(freqs, bandpass)

        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        emission = component.amp * bandpass_scaling
        input_unit = emission.unit
        unit_coefficient = get_bandpass_coefficient(
            freqs, bandpass, input_unit, output_unit
        )

        return emission * unit_coefficient


class PointSourceSimulation:
    """Simulation protocol for Point Source components."""

    def delta(
        self,
        component: PointSourceComponent,
        freq: Quantity,
        nside: int,
        fwhm: Quantity,
        output_unit: Union[str, Unit],
    ) -> Quantity:
        """See base class."""

        point_sources = component.amp * component.get_freq_scaling(
            freq, **component.spectral_parameters
        )

        emission = pointsources_to_healpix(
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

        bandpass = get_normalized_bandpass(freqs, bandpass)
        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        point_sources = component.amp * bandpass_scaling
        emission = pointsources_to_healpix(
            point_sources, component.catalog, nside, fwhm
        )
        input_unit = emission.unit
        unit_coefficient = get_bandpass_coefficient(
            freqs, bandpass, input_unit, output_unit
        )

        return emission * unit_coefficient


class LineSimulation:
    """Simulation protocol for Line components."""

    def delta(
        self,
        component: LineComponent,
        freq: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

    def bandpass(
        self,
        component: LineComponent,
        freqs: Quantity,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""


SIMULATION_PROTOCOLS = {
    DiffuseComponent: DiffuseSimulation(),
    PointSourceComponent: PointSourceSimulation(),
    LineComponent: LineSimulation(),
}


def get_simulation_protocol(component: SkyComponent) -> Simulation:
    """Returns a simulation protocol given a component."""

    for comp_type, protocol in SIMULATION_PROTOCOLS.items():
        if isinstance(component, comp_type):
            return protocol
    else:
        raise NotImplementedError(
            "simulation protocol not implemented for comp of type "
            f"{component.__class__.__name__}"
        )
