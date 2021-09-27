from typing import Dict, Iterator, List, Optional, Union

from astropy.units import Quantity, Unit
import healpy as hp

from cosmoglobe.sky.base_components import (
    SkyComponent,
    PointSourceComponent,
    DiffuseComponent,
    LineComponent,
)
from cosmoglobe.sky.simulator import SkySimulator, simulator
from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT, NO_SMOOTHING
from cosmoglobe.sky._exceptions import (
    NsideError,
    ComponentError,
    ComponentNotFoundError,
)


class SkyModel:
    r"""Sky model object representing the Cosmoglobe Sky Model.

    This class acts as a container for the various components making up
    the Cosmoglobe Sky Model, and provides methods to simulate the sky.
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
    >>> model = cosmoglobe.model_from_chain("path/to/chain", nside=256)
    >>> print(model)
    SkyModel(
      nside: 256
      components(
        (ame): AME(nu_p)
        (cmb): CMB()
        (dust): Dust(beta, T)
        (ff): FreeFree(Te)
        (radio): Radio(specind)
        (synch): Synchrotron(beta)
      )
    )

    Simulating the full sky emission at some frequency, given a beam
    FWHM:

    >>> import astropy.units as u
    >>> model(50*u.GHz, fwhm=30*u.arcmin)
    Smoothing point sources...
    Smoothing diffuse emission...
    [[ 2.25809786e+03  2.24380103e+03  2.25659060e+03 ... -2.34783682e+03
      -2.30464421e+03 -2.30387946e+03]
     [-1.64627550e+00  2.93583564e-01 -1.06788937e+00 ... -1.64354585e+01
       1.60621841e+01 -1.05506092e+01]
     [-4.15682825e+00  3.08881971e-01 -1.02012415e+00 ...  5.44745701e+00
      -4.71776995e+00  4.39850830e+00]] uK
    """

    simulator: SkySimulator = simulator

    def __init__(self, nside: int, components: List[SkyComponent]) -> None:
        """Initializes an instance of the Cosmoglobe Sky Model.

        Parameters
        ----------
        nside
            Healpix resolution of the maps in sky model.
        components
            A list of `SkyComponent`to include in the model.
        """

        self.nside = nside
        if not all(
            isinstance(
                component, (PointSourceComponent, DiffuseComponent, LineComponent)
            )
            for component in components
        ):
            raise ComponentError("all components must be subclasses of SkyComponent")

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
        components: Optional[List[str]] = None,
        fwhm: Quantity = NO_SMOOTHING,
        output_unit: Union[str, Unit] = DEFAULT_OUTPUT_UNIT,
    ) -> Quantity:
        r"""Simulates and returns the full sky model emission.

        Parameters
        ----------
        freqs
            A frequency, or a list of frequencies.
        bandpass
            Bandpass profile corresponding to the frequencies in `freqs`.
            If `bandpass` is None and `freqs` is a single frequency, a
            delta peak is assumed. Defaults to None.
        components
            List of component labels. If None, all components in the sky
            model is included. Defaults to None.
        fwhm
            The full width half max parameter of the Gaussian. Defaults to
            0.0, which indicates no smoothing of output maps.
        output_unit
            The desired output units of the emission. Units must be compatible
            with K_RJ or Jy/sr.

        Returns
        -------
        emission
            The simulated emission given the Cosmoglobe Sky Model.
        """

        if components is not None:
            if not all(component in self.components for component in components):
                raise ValueError("all component must be present in the model")
            components = [
                value for key, value in self.components.items() if key in components
            ]
        else:
            components = list(self.components.values())

        emission = self.simulator(
            self.nside, components, freqs, bandpass, fwhm, output_unit
        )

        return emission

    def remove_dipole(self, gal_cut: Quantity = 10 * Unit("deg")) -> None:
        """Removes the CMB dipole, from the CMB amp map."""

        if "cmb" not in self.components:
            raise ComponentNotFoundError("cmb component not present in model")

        hp.remove_dipole(
            self.components["cmb"].amp[0], gal_cut=gal_cut.to("deg").value, copy=False
        )

    def __iter__(self) -> Iterator:
        """Returns an iterator over the model components."""

        return iter(list(self.components.values()))

    def __repr__(self) -> str:
        """Representation of the SkyModel and all enabled components."""

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
