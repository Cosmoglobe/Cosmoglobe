from abc import ABC, abstractmethod
from typing import Dict, Protocol

from astropy.units import (
    Quantity,
    Unit,
    UnitConversionError,
    UnitsError,
    brightness_temperature,
    spectral,
)
import healpy as hp
import numpy as np

from cosmoglobe.sky import DEFAULT_OUTPUT_UNIT, DEFAULT_FREQ_UNIT
from cosmoglobe.sky.exceptions import NsideError


class SkyComponent(Protocol):
    """Protocol for a Cosmoglobe Sky Component."""

    @property
    def label(self) -> str:
        ...

    @property
    def amp(self) -> Quantity:
        ...

    @property
    def freq_ref(self) -> Quantity:
        ...

    @property
    def spectral_parameters(self) -> Dict[str, Quantity]:
        ...

    def __repr__(self) -> str:
        """Representation of a Cosmoglobe Sky Component."""


class DiffuseComponent(ABC):
    """Base class for diffuse emission sky components."""

    def __init__(
        self,
        label: str,
        amp: Quantity,
        freq_ref: Quantity,
        **spectral_parameters: Quantity,
    ) -> None:
        """Initializing base sky component class."""

        self._label = label
        self.freq_ref = freq_ref
        self.amp = amp
        self.spectral_parameters = spectral_parameters

    @abstractmethod
    def _get_freq_scaling(
        self, freqs: Quantity, **spectral_parameters: Quantity
    ) -> Quantity:
        """Returns the frequency scaling factors for a set of frequencies."""

    @property
    def label(self) -> str:
        """Component label."""

        return self._label

    @property
    def freq_ref(self) -> Quantity:
        """Reference frequency of the amplitude map of the sky component."""

        return self._freq_ref

    @freq_ref.setter
    def freq_ref(self, value: Quantity) -> None:
        """Validates the reference frequency data."""

        self._freq_ref = _validate_freq_ref(value)

    @property
    def amp(self) -> Quantity:
        """Amplitude map of the sky component at some reference frequency."""

        return self._amp

    @amp.setter
    def amp(self, value: Quantity) -> None:
        """Validates the amplitude data."""

        if not isinstance(value, Quantity):
            raise TypeError("ampltiude map must of type `astropy.units.Quantity`")

        try:
            hp.get_nside(value)
        except TypeError:
            raise NsideError(
                f"the number of pixels ({value.shape}) in the amplitude map of "
                f"{self.label} does not correspond to a valid HEALPIX nside"
            )

        if self.freq_ref is not None:
            if value.shape[0] != self.freq_ref.shape[0]:
                raise ValueError(
                    "shape of amplitude map must be either (1, `npix`) or "
                    "(3, `npix`) depending on if the component is polarized."
                )

            try:
                value = value.to(
                    DEFAULT_OUTPUT_UNIT,
                    equivalencies=brightness_temperature(self.freq_ref),
                )
            except UnitConversionError:
                raise UnitsError(
                    f"amplitude map must have units compatible with {DEFAULT_OUTPUT_UNIT}"
                )

        self._amp = value

    @property
    def spectral_parameters(self) -> Dict[str, Quantity]:
        """Dictionary containing spectral parameters for sky component."""

        return self._spectral_param

    @spectral_parameters.setter
    def spectral_parameters(self, values: Dict[str, Quantity]) -> None:
        """Validates the spectral parameter data."""

        spectral_parameters: Dict[str, Quantity] = {}
        for key, value in values.items():
            if not isinstance(value, Quantity):
                raise TypeError(
                    "spectral_parameter must be of type `astropy.units.Quantity`"
                )

            if value.ndim < 2 or value.shape[0] != self.amp.shape[0]:
                raise ValueError(
                    "shape of spectral parameter must be either (1, `npix`) or "
                    "(3, `npix`) if the parameter is a map, or (1, 1), (3, 1) "
                    "if the parameter is a scalar"
                )
            if value.shape[1] > 1:
                try:
                    hp.get_nside(value)
                except TypeError:
                    raise NsideError(
                        f"the number of pixels ({value.shape}) in the spectral "
                        f"parameter map {key} does not correspond to a valid "
                        "HEALPIX nside"
                    )
            spectral_parameters[key] = value

        self._spectral_param = spectral_parameters

    def __repr__(self) -> str:
        """Representation of the component."""

        return _sky_component_repr(self)


class PointSourceComponent(ABC):
    """Base class for PointSource emission sky components."""

    def __init__(
        self,
        label: str,
        catalog: np.ndarray,
        amp: Quantity,
        freq_ref: Quantity,
        **spectral_parameters: Quantity,
    ) -> None:
        """Initializing base sky component class."""

        self._label = label
        self._catalog = catalog
        self.freq_ref = freq_ref
        self.amp = amp
        self.spectral_parameters = spectral_parameters

    @abstractmethod
    def _get_freq_scaling(
        self, freqs: Quantity, **spectral_parameters: Quantity
    ) -> Quantity:
        """Returns the frequency scaling factors for a set of frequencies."""

    @property
    def label(self) -> str:
        """Component label."""

        return self._label

    @property
    def catalog(self) -> np.ndarray:
        """Catalog specifiying the angular coordinates of the source points."""

        return self._catalog

    @property
    def freq_ref(self) -> Quantity:
        """Reference frequency of the amplitude map of the sky component."""

        return self._freq_ref

    @freq_ref.setter
    def freq_ref(self, value: Quantity) -> None:
        """Validates the reference frequency data."""

        self._freq_ref = _validate_freq_ref(value)

    @property
    def amp(self) -> Quantity:
        """Amplitude map of the sky component at some reference frequency."""

        return self._amp

    @amp.setter
    def amp(self, value: Quantity) -> None:
        """Validates the amplitude data."""

        if not isinstance(value, Quantity):
            raise TypeError("ampltiude map must of type `astropy.units.Quantity`")

        if value.shape[0] != self.freq_ref.shape[0]:
            raise ValueError(
                "shape of amplitude map must be either (1, `npointsources`) or "
                "(3, `npointsources`) depending on if the component is polarized."
            )

        if value.shape[1] != self.catalog.shape[1]:
            raise ValueError(
                f"number of pointsources ({self.amp.shape[1]}) does not "
                f"match the number of cataloged points ({self.catalog.shape}). "
                "catalog shape must be (3, `npointsources`)"
            )

        try:
            (value / Unit("sr")).to(
                DEFAULT_OUTPUT_UNIT,
                equivalencies=brightness_temperature(self.freq_ref),
            )
        except UnitConversionError:
            raise UnitsError(
                "Radio sources must have units compatible with "
                f"{DEFAULT_OUTPUT_UNIT / Unit('sr')}"
            )

        self._amp = value

    @property
    def spectral_parameters(self) -> Dict[str, Quantity]:
        """Dictionary containing spectral parameters for sky component."""

        return self._spectral_param

    @spectral_parameters.setter
    def spectral_parameters(self, values: Dict[str, Quantity]) -> None:
        """Validates the spectral parameter data."""

        spectral_parameters: Dict[str, Quantity] = {}
        for key, value in values.items():
            if not isinstance(value, Quantity):
                raise TypeError(
                    "spectral_parameter must be of type `astropy.units.Quantity`"
                )

            if value.ndim < 2 or value.shape[0] != self.amp.shape[0]:
                raise ValueError(
                    "shape of spectral parameter must be either (1, `npointsources`) or "
                    "(3, `npointsources`) if the parameter is a map, or (1, 1), (3, 1) "
                    "if the parameter is a scalar"
                )

            spectral_parameters[key] = value

        self._spectral_param = spectral_parameters

    def __repr__(self) -> str:
        """Representation of the component."""

        return _sky_component_repr(self)


class LineComponent(ABC):
    """Base class for Line emission sky components."""

    def __init__(
        self,
        label: str,
        amp: Quantity,
        freq_ref: Quantity,
        **spectral_parameters: Quantity
    ) -> None:
        """Initializing base sky component class."""

        self._label = label
        self.freq_ref = freq_ref
        self.amp = amp
        self.spectral_parameters = spectral_parameters

    @property
    def label(self) -> str:
        """Component label."""

        return self._label

    @property
    def freq_ref(self) -> Quantity:
        """Reference frequency of the amplitude map of the sky component."""

        return self._freq_ref

    @freq_ref.setter
    def freq_ref(self, value: Quantity) -> None:
        """Validates the reference frequency data."""

        self._freq_ref = _validate_freq_ref(value)

    @property
    def amp(self) -> Quantity:
        """Amplitude map of the sky component at some reference frequency."""

        return self._amp

    @amp.setter
    def amp(self, value: Quantity) -> None:
        """Validates the amplitude data."""

        if not isinstance(value, Quantity):
            raise TypeError("ampltiude map must of type `astropy.units.Quantity`")

        if value.shape[0] != self.freq_ref.shape[0]:
            raise ValueError(
                "shape of amplitude map must be either (1, `npointsources`) or "
                "(3, `npointsources`) depending on if the component is polarized."
            )

        try:
            (value / Unit("km/s")).to(
                DEFAULT_OUTPUT_UNIT,
                equivalencies=brightness_temperature(self.freq_ref),
            )
        except UnitConversionError:
            raise UnitsError(
                "amplitude map must have units compatible with "
                f"{DEFAULT_OUTPUT_UNIT / Unit('km/s')}"
            )

        self._amp = value


def _validate_freq_ref(value: Quantity):
    """Validates the reference frequency data."""

    if not isinstance(value, Quantity):
        raise TypeError("reference frequency must of type `astropy.units.Quantity`")

    if value.shape not in ((1, 1), (3, 1)):
        raise ValueError(
            "shape of reference frequency must be either (1, 1) or "
            "(3, 1) depending on if the component is polarized."
        )

    try:
        value = value.to(DEFAULT_FREQ_UNIT, equivalencies=spectral())
    except UnitConversionError:
        raise UnitsError(
            f"reference frequency must have units compatible with {DEFAULT_FREQ_UNIT}"
        )

    return value


def _sky_component_repr(component: SkyComponent):
    main_repr = f"{component.__class__.__name__}"
    main_repr += "("
    extra_repr = ""
    for key in component.spectral_parameters.keys():
        extra_repr += f"{key}, "
    if extra_repr:
        extra_repr = extra_repr[:-2]
    main_repr += extra_repr
    main_repr += ")"

    return main_repr
