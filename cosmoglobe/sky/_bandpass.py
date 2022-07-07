from dataclasses import dataclass
from typing import Dict, List, Protocol, Union

from astropy.units import Quantity, Unit
from astropy.units.core import UnitBase
import numpy as np
from scipy.interpolate import RectBivariateSpline

from cosmoglobe.sky._intensity_derivatives import get_intensity_derivative
from cosmoglobe.sky._units import cmb_equivalencies


# Dict mapping number of grid points to the interpolation dimensin of the
# bandpass integration.
N_INTERPOLATION_GRID = {
    1: 1000,
    2: 100,
}


# @dataclass
# class Bandpass:
#     """Class representing a *normalized* bandpass."""

#     freqs: Quantity  # [Hz]
#     weights: Quantity  # [1/Hz]


def get_normalized_weights(
    freqs: Quantity,
    weights: Quantity,
    component_amp_unit: UnitBase,
) -> Quantity:
    """Normalizes a bandpass to units of unity under integration.

    Additionally, the bandpass is converted to the units of the amplitude
    map of the sky component used in the bandpass integration.

    Parameters
    ----------
    freqs
        Frequencies corresponding to bandpass weights.
    weights
        The weights of the bandpass.

    Returns
    -------
        Bandpass object.
    """

    # For pointsources, we need to divide the quantity by sr to make it
    # compatible with the cmb equivalencies.
    if component_amp_unit.is_equivalent("Jy"):
        component_amp_unit /= Unit("sr")

    weights = weights.to(component_amp_unit, equivalencies=cmb_equivalencies(freqs))
    weights /= np.trapz(weights, freqs)

    return weights


def get_bandpass_coefficient(
    freqs: Quantity,
    weights: Quantity,
    input_unit: Unit,
    output_unit: Unit,
) -> Quantity:
    """Returns the bandpass coefficient (Unit conversion factor for bandpass integrations).

    Parameters
    ----------
    freqs
        Frequencies corresponding to bandpass weights.
    weights
        The weights of the bandpass.
    input_unit
        The unit of the cosmoglobe data.
    ouput_unit
        The requested output unit of the simulated emission.

    Returns
    -------
    coefficient
        Bandpass coefficient representing the unitconversion between the
        input/ouput unit over the bandpass.
    """

    in_intensity_derivative = get_intensity_derivative(input_unit)
    out_intensity_derivative = get_intensity_derivative(output_unit)

    coefficient = np.trapz(weights * in_intensity_derivative(freqs), freqs) / np.trapz(
        weights * out_intensity_derivative(freqs), freqs
    )

    return coefficient


class FreqScalingFunc(Protocol):
    """Protocol defining the interface of the `get_freq_scaling` function of a SkyComponent."""

    def __call__(self, freqs: Quantity, **spectral_parameters: Quantity) -> Quantity:
        ...


class BandpassIntegration(Protocol):
    """Protocol defining the interface for a bandpass integration implementation."""

    def __call__(
        self,
        freqs: Quantity,
        weights: Quantity,
        freq_scaling_func: FreqScalingFunc,
        spectral_parameters: Dict[str, Quantity],
        *,
        interpolation_grid: Dict[str, Quantity],
    ) -> Union[float, List[float]]:
        """Computes the frequency scaling factor over bandpass integration.

        The bandpass integration is performed using the Commander mixing matrix
        formalism. Rather than computing the frequency scaling factor per pixel
        for each frequency in the bandpass, we grid out the spectral parameters
        of the component to a small range covering the spatial variations over
        IQU. We then compute the frequency scaling factor for each gridded
        parameter integrated over the bandpass and store these values. To
        estimate the bandpass integration factor, we then simply interpolate in
        these values.

        Parameters
        ----------
        freqs
            Frequencies corresponding to bandpass weights.
        weights
            The weights of the bandpass.
        freq_scaling_func
            Function that returns the SED scaling given frequencies and
            spectral parameters.
        spectral_parameters
            Dictionary containing the spectral parameters of the component we
            want to integrate over a bandpass.
        interpolation_grid
            Grid of spectral parameter values for the component. This grid
            used as interpolation values to compute the interpolated scaling
            factor.

        Returns
        -------
            Bandpass integration factor.
        """


def integrated_freq_scaling(
    freqs: Quantity,
    weights: Quantity,
    freq_scaling_func: FreqScalingFunc,
    spectral_parameters: Dict[str, Quantity],
    **_,
) -> float:
    """Bandpass integration 0D.

    The bandpass integration algorithm for a component with 0 spatially
    varying spectral parameters.
    """

    freq_scaling = freq_scaling_func(freqs, **spectral_parameters)
    scaling_factor = np.trapz(freq_scaling * weights, freqs)
    if np.ndim(scaling_factor) > 0:
        return np.expand_dims(scaling_factor, axis=1)

    return scaling_factor


def bandpass_interpolation_1d(
    freqs: Quantity,
    weights: Quantity,
    freq_scaling_func: FreqScalingFunc,
    spectral_parameters: Dict[str, Quantity],
    interpolation_grid: Dict[str, Quantity],
) -> List[float]:
    """Bandpass integration 1D

    The bandpass integration algorithm for a component with a single spatially
    varying spectral parameter."""

    # Grid dictionary only has one item
    ((key, spectral_parameter_grid),) = interpolation_grid.items()
    if spectral_parameters[key].shape[0] == 3:
        integrals = np.zeros((len(spectral_parameter_grid), 3))
    else:
        integrals = np.zeros((len(spectral_parameter_grid), 1))

    for idx, grid_point in enumerate(spectral_parameter_grid):
        scalar_params = {
            param: value for param, value in spectral_parameters.items() if param != key
        }
        freq_scaling = freq_scaling_func(freqs, **{key: grid_point}, **scalar_params)
        integrals[idx] = np.trapz(freq_scaling * weights, freqs)

    # We transpose the array to make it into row format similar to how
    # regular IQU maps are stored
    integrals = np.transpose(integrals)

    scaling = [
        np.interp(
            spectral_parameters[key][IQU].value,
            spectral_parameter_grid.value,
            integral,
        )
        for IQU, integral in enumerate(integrals)
    ]

    return scaling


def bandpass_interpolation_2d(
    freqs: Quantity,
    weights: Quantity,
    freq_scaling_func: FreqScalingFunc,
    spectral_parameters: Dict[str, Quantity],
    interpolation_grid: Dict[str, Quantity],
) -> List[float]:
    """Bandpass integration 2D

    The bandpass integration algorithm for a component with two spatially
    varying spectral parameters.
    """

    mesh_grid = {
        key: value
        for key, value in zip(
            interpolation_grid.keys(), np.meshgrid(*interpolation_grid.values())
        )
    }

    # Make n x n mesh grid for the spectral parameters
    n = len(list(interpolation_grid.values())[0])
    if any(
        spectral_param.shape[0] == 3 for spectral_param in spectral_parameters.values()
    ):
        integrals = np.zeros((n, n, 3))
    else:
        integrals = np.zeros((n, n, 1))

    for i in range(n):
        for j in range(n):
            grid_spectrals = {key: value[i, j] for key, value in mesh_grid.items()}
            freq_scaling = freq_scaling_func(freqs, **grid_spectrals)
            integrals[i, j] = np.trapz(freq_scaling * weights, freqs)
    integrals = np.transpose(integrals)

    scaling = []
    for IQU, integral in enumerate(integrals):
        f = RectBivariateSpline(*interpolation_grid.values(), integral)
        packed_spectrals = [
            spectral[IQU] if spectral.shape[0] == 3 else spectral[0]
            for spectral in spectral_parameters.values()
        ]
        scaling.append(f(*packed_spectrals, grid=False))

    return scaling


# Dict mapping number of the spatially varying spectral parameters to bandpass
# integration implementation.
BP_INTEGRATION_MAPPINGS: Dict[int, BandpassIntegration] = {
    0: integrated_freq_scaling,
    1: bandpass_interpolation_1d,
    2: bandpass_interpolation_2d,
}


def get_interpolation_grid(
    spectral_parameters: Dict[str, Quantity]
) -> Dict[str, Quantity]:
    """Returns a interpolation range.

    Computes the interpolation range of the spectral parameters of a
    sky component. We use a regular grid with n points for the range.

    Parameters
    ----------
    spectral_parameters
        Dictionary containing the spectral parameters of a given component.

    Returns
    -------
    interp_parameters
        Dictionary with a interpolation grid for each spatially varying
        spectral parameter.
    """

    dim = 0
    for spectral_parameter in spectral_parameters.values():
        if spectral_parameter.size > 3:
            dim += 1

    grid: Dict[str, Quantity] = {}
    if dim == 0:
        return grid

    try:
        n = N_INTERPOLATION_GRID[dim]
    except KeyError:
        raise NotImplementedError(
            "Bandpass integration for comps with more than two spectral "
            "parameters is not currently supported"
        )
    for key, value in spectral_parameters.items():
        if value.size > 3:
            grid_range = np.linspace(np.amin(value), np.amax(value), n)
            grid[key] = grid_range

    return grid


def get_bandpass_scaling(
    freqs: Quantity,
    weights: Quantity,
    freq_scaling_func: FreqScalingFunc,
    spectral_parameters: Dict[str, Quantity],
) -> Union[float, List[float]]:
    """Returns the appropriate Bandpass integration implementation given the spectral parameters."""

    grid = get_interpolation_grid(spectral_parameters)
    if (dim := len(grid)) not in BP_INTEGRATION_MAPPINGS:
        raise NotImplementedError(
            "Bandpass integration for comps with more than two spectral "
            "parameters is not currently supported"
        )

    return BP_INTEGRATION_MAPPINGS[dim](
        freqs=freqs,
        weights=weights,
        freq_scaling_func=freq_scaling_func,
        spectral_parameters=spectral_parameters,
        interpolation_grid=grid,
    )
