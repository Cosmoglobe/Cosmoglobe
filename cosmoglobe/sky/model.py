from __future__ import annotations
from typing import Dict, Iterator, List, Literal, Optional, Union

from astropy.units import Quantity, Unit
from astropy.units.core import UnitsError
import healpy as hp
import numpy as np

from cosmoglobe.sky._base_components import (
    DiffuseComponent,
    LineComponent,
    PointSourceComponent,
    SkyComponent,
)
from cosmoglobe.sky.components._labels import SkyComponentLabel
from cosmoglobe.sky._component_factory import get_components_from_chain
from cosmoglobe.sky._constants import (
    DEFAULT_OUTPUT_UNIT_STR,
    DEFAULT_BEAM_FWHM,
    DEFAULT_GAL_CUT,
)
from cosmoglobe.sky._exceptions import (
    NsideError,
    ComponentError,
    ComponentNotFoundError,
)
from cosmoglobe.sky.cosmoglobe import cosmoglobe_registry
from cosmoglobe.h5.chain import Chain


class SkyModel:
    r"""Sky model representing an initialized instance of the Cosmoglobe Sky Model.

    This class acts as a container for the various components making up
    the Cosmoglobe Sky Model and provides methods to simulate the sky.
    The primary use case of this class is to call its ``__call__``
    method, which simulates the sky at a single frequency :math:`\nu`,
    or integrated over a bandpass :math:`\tau`.

    Methods
    -------
    __call__
    remove_dipole

    Examples
    --------
    >>> import cosmoglobe
    >>> chain = cosmoglobe.get_test_chain()
    >>> model = cosmoglobe.SkyModel.from_chain(chain, nside=256)
    >>> print(model)
    SkyModel(
        version: BeyondPlanck
        nside: 256
        components(
            (ame): SpinningDust(freq_peak)
            (cmb): CMB()
            (dust): ModifiedBlackbody(beta, T)
            (ff): LinearOpticallyThin(T_e)
            (radio): AGNPowerLaw(alpha)
            (synch): PowerLaw(beta)
        )
    )

    Simulating the full sky emission at some frequency, given a beam
    FWHM:

    >>> import astropy.units as u
    >>> model(50*u.GHz, fwhm=30*u.arcmin)
    [[ 2.25809786e+03  2.24380103e+03  2.25659060e+03 ... -2.34783682e+03
      -2.30464421e+03 -2.30387946e+03]
     [-1.64627550e+00  2.93583564e-01 -1.06788937e+00 ... -1.64354585e+01
       1.60621841e+01 -1.05506092e+01]
     [-4.15682825e+00  3.08881971e-01 -1.02012415e+00 ...  5.44745701e+00
      -4.71776995e+00  4.39850830e+00]] uK_RJ
    """

    def __init__(
        self,
        nside: int,
        components: Dict[str, SkyComponent],
        version: Optional[str] = None,
    ) -> None:
        """Initializes an instance of the Cosmoglobe Sky Model.

        Parameters
        ----------
        nside
            Healpix resolution of the maps in sky model.
        components
            A list of pre-initialized `SkyComponent`to include in the model.
        version
            The version of the Cosmoglobe Model used in the sky model.
        """

        self.nside = nside
        self.components = components
        self.version = version

        if not all(
            isinstance(component, SkyComponent) for component in components.values()
        ):
            raise ComponentError("all components must be subclasses of SkyComponent")

        if not all(
            self.nside == hp.get_nside(component.amp)
            for component in components.values()
            if not isinstance(component, PointSourceComponent)
        ):
            raise NsideError(
                "all diffuse maps in the sky model needs to be at a common nside"
            )

    @classmethod
    def from_chain(
        cls,
        chain: Union[str, Chain],
        nside: int,
        components: Optional[List[str]] = None,
        model: str = "BeyondPlanck",
        samples: Union[range, int, Literal["all"]] = -1,
        burn_in: Optional[int] = None,
    ) -> SkyModel:
        """Initializes the SkyModel from a Cosmoglobe chain.

        Parameters
        ----------
        chain
            Path to a Cosmoglobe chainfile or a Chain object.
        nside
            Model HEALPIX map resolution parameter.
        components
            List of components to include in the model.
        model
            String representing which sky model to use. Defaults to BeyondPlanck.
        samples
            The sample number for which to extract the model. If the input
            is 'all', then the model will an average of all samples in the chain.
            Defaults to the last sample in the chain.
        burn_in
            Burn in sample for which all previous samples are disregarded.

        Returns
        -------
        sky_model
            Initialized sky model.
        """

        cosmoglobe_model = cosmoglobe_registry.get_model(model)
        initialized_components = get_components_from_chain(
            chain=chain,
            nside=nside,
            components=components,
            model=cosmoglobe_model,
            samples=samples,
            burn_in=burn_in,
        )

        return cls(
            nside=nside,
            components=initialized_components,
            version=cosmoglobe_model.version,
        )

    @classmethod
    def from_hub(cls) -> SkyModel:
        """Initializes the SkyModel from a public cosmoglobe data release."""

        raise NotImplementedError

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

    def __call__(
        self,
        freqs: Quantity,
        weights: Optional[Quantity] = None,
        *,
        components: Optional[List[str]] = None,
        fwhm: Quantity = DEFAULT_BEAM_FWHM,
        output_unit: str = DEFAULT_OUTPUT_UNIT_STR,
    ) -> Quantity:
        r"""Simulates and returns the full sky model emission.

        Parameters
        ----------
        freqs
            A frequency, or a list of frequencies corresponding to the 
            bandpass weights.
        weights
            Bandpass weights corresponding to the frequencies in `freqs`.
            The units of this quantity must match the units of the related 
            detector, i.e, they must be compatible with K_RJ, K_CMB, or Jy/sr.
            Even if the bandpass weights are already normalized and now have 
            units of 1/Hz, redefine the quantities unites to be that of the 
            detector. If `bandpass` is None and `freqs` is a single frequency, 
            a delta peak is assumed. Defaults to None.
        components
            List of component labels. If None, all components in the sky
            model is included. Defaults to None.
        fwhm
            The full width half max parameter of the Gaussian. Defaults to
            0.0, which indicates no smoothing of output maps.
        output_unit
            The output units of the emission. For instance 'uK_RJ', 'uK_CMB', 
            or 'MJy/sr'. If Kelvin, Rayleigh-Jeans or CMB must be specified.

        Returns
        -------
        emission
            The simulated emission given the Cosmoglobe Sky Model.
        """

        if components is None:
            included_components = list(self.components.values())
        else:
            if not all(component in self.components for component in components):
                raise ValueError("all component must be present in the model")
            included_components = [
                component
                for label, component in self.components.items()
                if label in components
            ]

        diffuse_components: List[Union[DiffuseComponent, LineComponent]] = []
        pointsource_components: List[PointSourceComponent] = []
        for component_class in included_components:
            if isinstance(component_class, (DiffuseComponent, LineComponent)):
                diffuse_components.append(component_class)
            elif isinstance(component_class, PointSourceComponent):
                pointsource_components.append(component_class)

        emission_unit = Unit(output_unit)
        emission = Quantity(np.zeros((3, hp.nside2npix(self.nside))), unit=emission_unit)

        for diffuse_component in diffuse_components:
            component_emission = diffuse_component.simulate_emission(
                freqs=freqs,
                weights=weights,
                nside=self.nside,
                fwhm=fwhm,
                output_unit=emission_unit,
            )
            for IQU, diffuse_emission in enumerate(component_emission):
                emission[IQU] += diffuse_emission

        # We smooth all diffuse components together in a single smoothing operation.
        if fwhm != DEFAULT_BEAM_FWHM:
            emission = Quantity(
                hp.smoothing(emission, fwhm=fwhm.to("rad").value), unit=emission.unit
            )

        # Pointsource emissions are already smoothed during the stage where 
        # each source is mapped to the HEALPIX map.
        for pointsource_component in pointsource_components:
            component_emission = pointsource_component.simulate_emission(
                freqs=freqs,
                weights=weights,
                nside=self.nside,
                fwhm=fwhm,
                output_unit=emission_unit,
            )
            for IQU, pointsource_emission in enumerate(component_emission):
                emission[IQU] += pointsource_emission

        return emission

    def remove_dipole(
        self,
        gal_cut: Quantity = DEFAULT_GAL_CUT,
        return_dipole: bool = False,
    ) -> Optional[Quantity]:
        """Removes the Solar dipole (and monopole), from the CMB sky component.

        Parameters
        ----------
        gal_cut
            Ignores pixels in range [-gal_cut;+gal_cut] when fitting and
            subtracting the dipole.
        return_dipole
            If True, returns the dipole map.

        Returns
        _______
        dipole
            The dipole of the CMB sky component.
        """

        if not any(
            component.label == SkyComponentLabel.CMB
            for component in self.components.values()
        ):
            raise ComponentNotFoundError("CMB component not found in sky model.")

        try:
            gal_cut = gal_cut.to("deg")
        except UnitsError:
            raise UnitsError(
                "gal_cut must be an astropy quantity compatible with 'deg'"
            )

        if return_dipole:
            dipole_subtracted_amp = Quantity(
                hp.remove_dipole(
                    self["cmb"].amp[0],
                    gal_cut=gal_cut.to("deg").value,
                ),
                unit=self["cmb"].amp.unit,
            )
            dipole = self["cmb"].amp[0] - dipole_subtracted_amp
            self["cmb"].amp[0] = dipole_subtracted_amp

            return dipole

        hp.remove_dipole(
            self["cmb"].amp[0], gal_cut=gal_cut.to("deg").value, copy=False
        )

    def __iter__(self) -> Iterator:
        """Returns an iterator over the model components."""

        return iter(list(self.components.values()))

    def __getitem__(self, key: str) -> SkyComponent:
        """Returns a SkyComponent class."""

        try:
            return self.components[key]
        except KeyError:
            raise ComponentNotFoundError(f"component {key} not found in sky model.")

    def __repr__(self) -> str:
        """Representation of the SkyModel and all enabled components."""

        reprs = []
        for label, component in self.components.items():
            component_repr = repr(component) + "\n"
            reprs.append(f"({label}): {component_repr}")

        main_repr = "SkyModel("
        if self.version is not None:
            main_repr += f"\n  version: {self.version}"
        main_repr += f"\n  nside: {self._nside}"
        main_repr += "\n  components( "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += "  )"
        main_repr += "\n)"

        return main_repr
