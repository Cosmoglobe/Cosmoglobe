from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import warnings

from astropy.units import (
    quantity_input,
    Quantity,
    spectral,
    Unit,
    UnitConversionError,
    UnitsError,
)
import healpy as hp
import numpy as np

from cosmoglobe.sky._constants import (
    DEFAULT_OUTPUT_UNIT,
    DEFAULT_BEAM_FWHM,
    SIGNAL_UNITS,
)
from cosmoglobe.sky._exceptions import NsideError
from cosmoglobe.sky._bandpass import (
    get_normalized_weights,
    get_bandpass_coefficient,
    get_bandpass_scaling,
)
from cosmoglobe.sky._beam import pointsources_to_healpix
from cosmoglobe.sky._freq_range import FrequencyRange
from cosmoglobe.sky._units import cmb_equivalencies
from cosmoglobe.sky.components._labels import SkyComponentLabel


class SkyComponent(ABC):
    """Abstract base class for sky components.

    Attributes
    ----------
    label
        Name of the sky component.
    amp
        Amplitude map.
    freq_ref
        Reference frequency of the amplitude map.
    spectral_parameters
        Dictionary containing the spectral parameters for the component.
    """

    label: SkyComponentLabel
    freq_range: Optional[Tuple[Quantity, Quantity]] = None

    def __init__(
        self,
        amp: Quantity,
        freq_ref: Quantity,
        **spectral_parameters: Quantity,
    ):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = spectral_parameters

        self._validate_freq_ref(freq_ref)

        if self.freq_range is not None:
            self._freq_range = FrequencyRange(*self.freq_range)

    @abstractmethod
    def get_delta_emission(
        self,
        freq: Quantity,
        *,
        nside: Optional[int],
        fwhm: Quantity,
        output_unit: Unit,
    ) -> Quantity:
        """Computes and returns the emission for a delta frequency.

        Parameters
        ----------
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

    @abstractmethod
    def get_bandpass_emission(
        self,
        freqs: Quantity,
        weights: Optional[Quantity],
        *,
        nside: Optional[int],
        fwhm: Quantity,
        output_unit: Unit,
    ) -> Quantity:
        """Computes and returns the emission for a bandpass profile.

        Parameters
        ----------
        freqs
            An array of frequencies for which to compute the bandpass emission
            over.
        weights
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
            Bandpass integrated emission.
        """

    @quantity_input(
        freqs="Hz",
        weights=("K_RJ", "K_CMB", "Jy/sr", None),
        fwhm=("deg", "arcmin", "rad"),
    )
    def simulate_emission(
        self,
        freqs: Quantity,
        weights: Optional[Quantity] = None,
        nside: Optional[int] = None,
        fwhm: Quantity = DEFAULT_BEAM_FWHM,
        output_unit: Unit = DEFAULT_OUTPUT_UNIT,
    ) -> Quantity:
        """Returns the simulated emission for a sky component.

        Parameters
        ----------
        freqs
            An array of frequencies for which to compute the bandpass emission
            over.
        weights
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
            Bandpass integrated emission.
        """

        if not any(unit.is_equivalent(output_unit) for unit in SIGNAL_UNITS):
            raise UnitsError(
                "output unit must be one of "
                f"{', '.join(unit.to_string() for unit in SIGNAL_UNITS)}"
            )

        # If the emission from the sky component is negligable at the
        # frequencies given by `freqs`, we simply return 0.
        if self.freq_range is not None and freqs not in self._freq_range:
            return Quantity([0, 0, 0], unit=output_unit)

        if freqs.size > 1:
            if weights is not None and freqs.shape != weights.shape:
                raise ValueError(
                    "bandpass frequencies and weights must have the same shape"
                )

            if weights is None:
                warnings.warn(
                    "bandpass weights are None. Defaulting to top-hat weights."
                )
                weights = np.ones(freqs_len := len(freqs)) / freqs_len * Unit("K_RJ")

            return self.get_bandpass_emission(
                freqs=freqs,
                weights=weights,
                output_unit=output_unit,
                fwhm=fwhm,
                nside=nside,
            )

        return self.get_delta_emission(
            freq=freqs,
            output_unit=output_unit,
            fwhm=fwhm,
            nside=nside,
        )

    @property
    def is_polarized(self) -> bool:
        """Returns True if component emits polarized signal and False otherwise."""

        return self.amp.shape[0] == 3

    @staticmethod
    def _validate_freq_ref(freq_ref: Quantity):
        """Validates the type and shape of a reference frequency attribute."""

        if not isinstance(freq_ref, Quantity):
            raise TypeError("reference frequency must of type `astropy.units.Quantity`")

        if freq_ref.shape not in ((1, 1), (3, 1)):
            raise ValueError(
                "shape of reference frequency must be either (1, 1) or "
                "(3, 1) depending on if the component is polarized."
            )

        try:
            freq_ref.to("Hz", equivalencies=spectral())
        except UnitConversionError:
            raise UnitsError("reference frequency must have units compatible with Hz")

    def __repr__(self) -> str:
        """Representation of the sky component."""

        main_repr = f"{self.__class__.__name__}"
        main_repr += "("
        extra_repr = ""
        for key in self.spectral_parameters.keys():
            extra_repr += f"{key}, "
        if extra_repr:
            extra_repr = extra_repr[:-2]
        main_repr += extra_repr
        main_repr += ")"

        return main_repr


class DiffuseComponent(SkyComponent):
    """Base class for diffuse sky components.

    A diffuse sky component must implement the `get_freq_scaling` method
    which computes the unscaled component SED.
    """

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        # All input quantities are validated to make sure they can be
        # properly broadcasted under the sky simulation.
        self._validate_amp(amp)
        self._validate_spectral_parameters(spectral_parameters)

    @abstractmethod
    def get_freq_scaling(
        self, freqs: Quantity, **spectral_parameters: Quantity
    ) -> Quantity:
        """Computes and returns the frequency scaling factor.

        Parameters
        ----------
        freqs
            A frequency, or a list of frequencies for which to compute the
            scaling factor.
        **spectral_parameters
            The (unpacked) spectral parameters contained in the
            spectral_parameters dictionary for a given component.

        Returns
        -------
        scaling_factor
            The factor with which to scale the amplitude map for a set of
            frequencies.
        """

    def get_delta_emission(self, freq: Quantity, output_unit: Unit, **_) -> Quantity:
        """See base class."""

        emission = self.amp * self.get_freq_scaling(freq, **self.spectral_parameters)

        return emission.to(output_unit, equivalencies=cmb_equivalencies(freq))

    def get_bandpass_emission(
        self, freqs: Quantity, weights: Quantity, output_unit: Unit, **_
    ) -> Quantity:
        """See base class."""

        normalized_weights = get_normalized_weights(
            freqs=freqs,
            weights=weights,
            component_amp_unit=self.amp.unit,
        )

        bandpass_scaling = get_bandpass_scaling(
            freqs=freqs,
            weights=normalized_weights,
            freq_scaling_func=self.get_freq_scaling,
            spectral_parameters=self.spectral_parameters,
        )
        emission = self.amp * bandpass_scaling
        unit_coefficient = get_bandpass_coefficient(
            freqs=freqs,
            weights=normalized_weights,
            input_unit=emission.unit,
            output_unit=output_unit,
        )
        emission *= unit_coefficient

        # Converts to the correct prefix
        if emission != output_unit:
            emission = emission.to(output_unit)

        return emission

    def _validate_amp(self, amp: Quantity) -> None:
        if not isinstance(amp, Quantity):
            raise TypeError("ampltiude map must of type `astropy.units.Quantity`")

        try:
            hp.get_nside(amp)
        except TypeError:
            raise NsideError(
                f"the number of pixels ({amp.shape}) in the amplitude map "
                "does not correspond to a valid HEALPIX nside"
            )

        if self.freq_ref is not None:
            if amp.shape[0] != self.freq_ref.shape[0]:
                raise ValueError(
                    "shape of amplitude map must be either (1, `npix`) or "
                    "(3, `npix`) depending on if the component is polarized."
                )

            try:
                amp.to(
                    DEFAULT_OUTPUT_UNIT,
                    equivalencies=cmb_equivalencies(self.freq_ref),
                )
            except UnitConversionError:
                raise UnitsError(
                    f"amplitude map must have units compatible with {DEFAULT_OUTPUT_UNIT}"
                )

    def _validate_spectral_parameters(
        self, spectral_parameters: Dict[str, Quantity]
    ) -> None:
        for name, parameter in spectral_parameters.items():
            if not isinstance(parameter, Quantity):
                raise TypeError(
                    "spectral_parameter must be of type `astropy.units.Quantity`"
                )

            if parameter.ndim < 2 or parameter.shape[0] != self.amp.shape[0]:
                raise ValueError(
                    "shape of spectral parameter must be either (1, `npix`) or "
                    "(3, `npix`) if the parameter is a map, or (1, 1), (3, 1) "
                    "if the parameter is a scalar"
                )
            if parameter.shape[1] > 1:
                try:
                    hp.get_nside(parameter)
                except TypeError:
                    raise NsideError(
                        f"the number of pixels ({parameter.shape}) in the spectral "
                        f"parameter map {name} does not correspond to a valid "
                        "HEALPIX nside"
                    )


class PointSourceComponent(SkyComponent):
    """Base class for PointSource sky components.

    A pointsource sky component must implement the `get_freq_scaling` method
    which computes the unscaled component SED.

    Additionally, a catalog mapping each source to a coordinate given by
    latitude and longitude must be specificed as a class/instance attribute.
    """

    catalog: Quantity

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        # All input quantities are validated to make sure they can be
        # properly broadcasted under the sky simulation.
        self._validate_amp(amp)
        try:
            self._validate_catalog(self.catalog)
        except AttributeError:
            raise AttributeError(
                "a point source catalog must be specified as a class attribute."
            )
        self._validate_spectral_parameters(spectral_parameters)

    @abstractmethod
    def get_freq_scaling(
        self, freqs: Quantity, **spectral_parameters: Quantity
    ) -> Quantity:
        """Computes and returns the frequency scaling factor.

        Parameters
        ----------
        freqs
            A frequency, or a list of frequencies for which to compute the
            scaling factor.
        **spectral_parameters
            The (unpacked) spectral parameters contained in the
            spectral_parameters dictionary for a given component.

        Returns
        -------
        scaling_factor
            The factor with which to scale the amplitude map for a set of
            frequencies.
        """

    def get_delta_emission(
        self, freq: Quantity, nside: int, fwhm: Quantity, output_unit: Unit
    ) -> Quantity:
        """See base class."""

        scaled_point_sources = self.amp * self.get_freq_scaling(
            freq, **self.spectral_parameters
        )
        emission = pointsources_to_healpix(
            point_sources=scaled_point_sources,
            catalog=self.catalog,
            nside=nside,
            fwhm=fwhm,
        )

        return emission.to(output_unit, equivalencies=cmb_equivalencies(freq))

    def get_bandpass_emission(
        self,
        freqs: Quantity,
        weights: Quantity,
        nside: int,
        fwhm: Quantity,
        output_unit: Unit,
    ) -> Quantity:
        """See base class."""

        normalized_weights = get_normalized_weights(
            freqs=freqs,
            weights=weights,
            component_amp_unit=self.amp.unit,
        )
        bandpass_scaling = get_bandpass_scaling(
            freqs=freqs,
            weights=normalized_weights,
            freq_scaling_func=self.get_freq_scaling,
            spectral_parameters=self.spectral_parameters,
        )

        scaled_point_sources = self.amp * bandpass_scaling
        emission = pointsources_to_healpix(
            point_sources=scaled_point_sources,
            catalog=self.catalog,
            nside=nside,
            fwhm=fwhm,
        )
        input_unit = emission.unit
        unit_coefficient = get_bandpass_coefficient(
            freqs=freqs,
            weights=normalized_weights,
            input_unit=input_unit,
            output_unit=output_unit,
        )
        emission *= unit_coefficient

        # Converts to the correct prefix
        if emission != output_unit:
            emission = emission.to(output_unit)

        return emission

    def _validate_amp(self, amp: Quantity) -> None:
        if not isinstance(amp, Quantity):
            raise TypeError("ampltiude map must of type `astropy.units.Quantity`")

        if amp.shape[0] != self.freq_ref.shape[0]:
            raise ValueError(
                "shape of amplitude map must be either (1, `npointsources`) or "
                "(3, `npointsources`) depending on if the component is polarized."
            )
        try:
            (amp / Unit("sr")).to(
                DEFAULT_OUTPUT_UNIT,
                equivalencies=cmb_equivalencies(self.freq_ref),
            )
        except UnitConversionError:
            raise UnitsError(
                "Radio sources must have units compatible with "
                f"{DEFAULT_OUTPUT_UNIT / Unit('sr')}"
            )

    def _validate_spectral_parameters(
        self, spectral_parameters: Dict[str, Quantity]
    ) -> None:
        for parameter in spectral_parameters.values():
            if not isinstance(parameter, Quantity):
                raise TypeError(
                    "spectral_parameter must be of type `astropy.units.Quantity`"
                )

            if parameter.ndim < 2 or parameter.shape[0] != self.amp.shape[0]:
                raise ValueError(
                    "shape of spectral parameter must be either (1, `npointsources`) or "
                    "(3, `npointsources`) if the parameter is a map, or (1, 1), (3, 1) "
                    "if the parameter is a scalar"
                )

    def _validate_catalog(self, catalog: Quantity):
        if catalog.shape[1] != self.amp.shape[1]:
            raise ValueError(
                f"number of pointsources ({self.amp.shape[1]}) does not "
                f"match the number of cataloged points ({self.catalog.shape}). "
                "catalog shape must be (3, `npointsources`)"
            )


class LineComponent(SkyComponent):
    """Base class for Line emission sky components.

    NOTE: This class is still in development, and no Line-emission components
    are currently implemented in Cosmoglobe.
    """

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        self._validate_amp(amp)

    def get_delta_emission(
        self,
        freq: Quantity,
        output_unit: Unit,
        **_,
    ) -> Quantity:
        """See base class."""

    def get_bandpass_emission(
        self,
        freqs: Quantity,
        weights: Quantity,
        output_unit: Unit,
        **_,
    ) -> Quantity:
        """See base class."""

    def _validate_amp(self, value: Quantity) -> None:
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
                equivalencies=cmb_equivalencies(self.freq_ref),
            )
        except UnitConversionError:
            raise UnitsError(
                "amplitude map must have units compatible with "
                f"{DEFAULT_OUTPUT_UNIT / Unit('km/s')}"
            )
