from typing import Dict, Iterator, List, Optional, Union

from astropy.units import Quantity, Unit
import healpy as hp
import numpy as np

from cosmoglobe.sky import DEFAULT_OUTPUT_UNIT, NO_SMOOTHING
from cosmoglobe.sky.basecomponent import (
    SkyComponent,
    PointSourceComponent,
    DiffuseComponent,
    LineComponent,
)
from cosmoglobe.sky.exceptions import NsideError, SkyModelComponentError
from cosmoglobe.sky.simulation import SkySimulator, simulator


class SkyModel:
    """Cosmoglobe Sky Model."""

    simulator: SkySimulator = simulator

    def __init__(self, nside: int, components: List[SkyComponent]) -> None:
        """Initializes the Sky Model."""

        self.nside = nside
        if not all(
            isinstance(
                component, (PointSourceComponent, DiffuseComponent, LineComponent)
            )
            for component in components
        ):
            raise SkyModelComponentError(
                "all components must be subclasses of SkyComponent"
            )

        if not all(
            self.nside == hp.get_nside(component.amp)
            for component in components
            if not isinstance(component, PointSourceComponent)
        ):
            raise NsideError(
                "all diffuse maps in the sky model needs to be at a common nside"
            )
        self._components = {component.label: component for component in components}

    @property
    def nside(self) -> int:
        """HEALPIX map resolution of the maps in the Sky Model."""

        return self._nside

    @nside.setter
    def nside(self, value: int) -> None:
        """Validating the nside."""

        try:
            if not hp.isnsideok(value, nest=True):
                raise NsideError("nside needs to be a power of 2")
        except (TypeError, ValueError):
            raise TypeError("nside must be an integer")

        self._nside = int(value)

    @property
    def components(self) -> Dict[str, SkyComponent]:
        """Sky Components in the model."""

        return self._components

    def __call__(
        self,
        freqs: Quantity,
        bandpass: Optional[Quantity] = None,
        *,
        comps: List[str] = None,
        fwhm: Quantity = NO_SMOOTHING,
        output_unit: Union[str, Unit] = DEFAULT_OUTPUT_UNIT,
    ) -> Quantity:
        """Simulates the diffuse sky emission."""

        if comps is not None:
            if not all(component in self.components for component in comps):
                raise ValueError("all component must be present in the model")
            components = [
                value for key, value in self.components.items() if key in comps
            ]
        else:
            components = list(self.components.values())

        emission = self.simulator(
            self.nside, components, freqs, bandpass, fwhm, output_unit
        )

        return emission

    def __iter__(self) -> Iterator:
        """Returns an iterator with active model components"""

        return iter(list(self.components.values()))

    def __repr__(self) -> str:
        """Representation of the Model and all enabled components."""

        reprs = []
        for label, component in self.components.items():
            component_repr = repr(component) + "\n"
            reprs.append(f"({label}): {component_repr}")

        main_repr = f"SkyModel("
        main_repr += f"\n  nside: {self._nside}"
        main_repr += "\n  components( "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += f"  )"
        main_repr += f"\n)"

        return main_repr
