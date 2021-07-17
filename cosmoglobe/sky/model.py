from cosmoglobe.sky import components
from cosmoglobe.utils import utils
from cosmoglobe.sky.base import (
    _Component, 
    _DiffuseComponent, 
    _PointSourceComponent,
    _LineComponent
)

from typing import List, Tuple
import astropy.units as u
import healpy as hp
import numpy as np


class Model:
    r"""Class that interfaces the Cosmoglobe Sky Model with commander3 
    outputs for the purpose of producing astrophysical sky maps.

    This class acts as a container for the various components making up 
    the Cosmoglobe Sky Model, and provides methods to simulate the sky. 
    The primary use case of this class is to call its ``__call__`` 
    function, which simulates the sky at a single frequency :math:`\nu`, 
    or integrated over a bandpass :math:`\tau` given the present 
    components in the model.
    
    Attributes
    ----------
    AME : `cosmoglobe.sky.components.AME`
        The AME sky component.
    CMB : `cosmoglobe.sky.components.CMB`
        The CMB sky component.
    dust : `cosmoglobe.sky.components.Dust`
        The dust sky component.
    ff : `cosmoglobe.sky.components.FreeFree`
        The free-free sky component.
    radio : `cosmoglobe.sky.components.Radio`
        The radio sky component.
    synch : `cosmoglobe.sky.components.Synchrotron`
        The synchrotron sky component.
    nside : int
        Healpix resolution of the maps in sky model.
    components : dict
        Dictionary of all sky components included in the model.
    is_polarized : bool

    Methods
    -------
    __call__
    disable
    enable
    to_nside

    See Also
    --------
    __call__ : 
        Simulates the sky at a given frequency :math:`\nu` or over a 
        bandpass :math:`\tau` given the Cosmoglobe Sky Model.
    """

    def __init__(
        self, 
        components: List[_Component] = None, 
        nside: int = None
    ) -> None:
        """Initializing a sky model. 

        Parameters
        ----------
        components : list, optional
            A list of `cosmoglobe.sky.component.Component` objects that 
            constitutes the sky model (by default this is None and the 
            components are iteratively added as they are read from a 
            commander3 chain).
        nside : int, optional
            Healpix resolution of the maps in sky model (the default is 
            None, in which the model automatically detects the nside from
            the components).
        """

        self.nside = nside
        self.components = {}
        self.disabled_components = {}

        if components is not None:
            for component in components:
                self._add_component(component)


    def _add_component(self, component):
        if not issubclass(component.__class__, _Component):
            raise TypeError(
                f'{component} is not a subclass of cosmoglobe.sky._Component'
            )

        name = component.label
        if name in self.components:
            raise KeyError(f'component {name} already exists in model')

        if isinstance(component, _PointSourceComponent):
            if not hasattr(components, 'nside'):
                component.nside = self.nside
        else:
            nside = hp.get_nside(component.amp)
            if nside != self.nside:
                if self.nside is None:
                    self.nside = nside
                else:
                    raise ValueError(
                        f'component {name!r} has a reference map at NSIDE='
                        f'{nside}, but model NSIDE is set to {self.nside}'
                    )

        setattr(self, name, component)
        self.components[name] = component


    @u.quantity_input(
        freq=u.Hz, 
        bandpass=(u.Jy/u.sr, u.K, None), 
        fwhm=(u.rad, u.deg, u.arcmin)
    )
    def __call__(
        self, 
        freqs, 
        bandpass=None, 
        fwhm=0.0 * u.rad, 
        output_unit: Tuple[u.UnitBase, str] = u.uK
    ):
        r"""Computes the model emission (sum of all component emissions) 
        at a single frequency :math:`\nu` or integrated over a bandpass :math:`\tau`.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            A frequency, or a list of frequencies for which to evaluate the
            sky emission.
        bandpass : `astropy.units.Quantity`, optional
            Bandpass profile corresponding to the frequencies. Default is 
            None. If `bandpass` is None and `freqs` is a single frequency,
            a delta peak is assumed (unless `freqs` is a list of 
            frequencies, for which a top-hat bandpass is used to perform  
            bandpass integration instead).
        fwhm : `astropy.units.Quantity`, optional
            The full width half max parameter of the Gaussian (Default is 
            0.0, which indicates no smoothing of output maps).
        output_unit : `astropy.units.Unit`, optional
            The desired output units of the emission (By default the 
            output unit of the model is always in 
            :math:`\mathrm{\mu K_{RJ}}`.

        Returns
        -------
        `astropy.units.Quantity`
            The full model emission.

        Notes
        -----
        This function computes the following expression (assuming that 
        all default Cosmoglobe Sky components are present in the model):

        .. math::

            \boldsymbol{s}_{\mathrm{RJ}}(\nu) &=\boldsymbol{a}_{\mathrm{CMB}}
             \frac{x^{2} \mathrm{e}^{x}}{\left(\mathrm{e}^{x}-1\right)^{2}} 
            \frac{\left(\mathrm{e}^{x_{0}}-1\right)^{2}}{x_{0}^{2} 
            \mathrm{e}^{x_{0}}}\\
            &+\boldsymbol{a}_{\mathrm{s}}\left(\frac{\nu}{\nu_{0, 
            \mathrm{~s}}}\right)^{\beta_{\mathrm{s}}}\\
            &+\boldsymbol{a}_{\mathrm{ff}} \frac{g_{\mathrm{ff}}
            \left(\nu ; T_{e}\right)}{g_{\mathrm{ff}}\left(\nu_{0, 
            \mathrm{ff}} ; T_{e}\right)}\left(\frac{\nu_{0, 
            \mathrm{ff}}}{\nu}\right)^{2}\\
            &+\boldsymbol{a}_{\mathrm{sd}}\left(\frac{\nu_{0, 
            \mathrm{sd}}}{\nu}\right)^{2} \frac{s_{0}^{\mathrm{sd}}
            \left(\nu \cdot \frac{\nu_{p}}{30.0\; \mathrm{GHz}}\right)}
            {s_{0}^{\mathrm{sd}}\left(\nu_{0, \mathrm{sd}} \cdot 
            \frac{\nu_{p}}{30.0\; \mathrm{GHz}}\right)} \\
            &+\boldsymbol{a}_{\mathrm{d}}\left(\frac{\nu}{\nu_{0, 
            \mathrm{~d}}}\right)^{\beta_{\mathrm{d}}+1} 
            \frac{\mathrm{e}^{h \nu_{0, \mathrm{~d}} 
            / k T_{\mathrm{d}}}-1}{\mathrm{e}^{\mathrm{h} \nu 
            / k T_{\mathrm{d}}}-1}+\\&+\sum_{j=1}^{N_{\mathrm{src}}} 
            \boldsymbol{a}_{\mathrm{src}}^{j}\left(\frac{\nu}{\nu_{0, 
            \mathrm{src}}}\right)^{\alpha_{j, \mathrm{src}}-2}.

        For more information on the current implementation of the 
        Cosmoglobe Sky Model, see `BeyondPlanck (2020), Section 3.6 
        <https://arxiv.org/pdf/2011.05609.pdf>`_.

        Examples
        --------
        Full Sky Model emission at :math:`50\; \mathrm{GHz}`:

        >>> from cosmoglobe.sky import skymodel
        >>> import astropy.units as u
        >>> model = skymodel(nside=256)
        >>> model(50*u.GHz)[0]  # Stokes I parameter
        [ 2234.74893115  2291.99921295  2323.98779311 ... -2320.74732945
         -2271.54465982 -2292.22248419] uK

        Dust emission at :math:`500\; \mathrm{GHz}` smoothed with a 
        :math:`50\; '`  Gaussian beam in units of 
        :math:`\mathrm{MJy} / \mathrm{sr}`:

        >>> model.dust(500*u.GHz, fwhm=50*u.arcmin, output_unit='MJy/sr')[0]
        [3.08898797e-05 3.25889729e-05 3.76313847e-05 ... 3.76599942e-05
         3.99570235e-05 4.46317102e-05] MJy / sr
        """

        if self.is_polarized:
            shape = (3, hp.nside2npix(self.nside))
        else:
            shape = (1, hp.nside2npix(self.nside))
        diffuse_emission = np.zeros(shape)
        ptsrc_emission = np.zeros(shape)

        unit = utils._str_to_astropy_unit(output_unit)
        diffuse_emission = u.Quantity(diffuse_emission, unit=unit)
        ptsrc_emission = u.Quantity(ptsrc_emission, unit=unit)

        for comp in self:
            if isinstance(comp, _DiffuseComponent):
                comp_emission = comp(freqs, bandpass, output_unit=output_unit)
                for idx, row in enumerate(comp_emission):
                    diffuse_emission[idx] += row

            elif isinstance(comp, _PointSourceComponent):
                comp_emission = comp(
                    freqs, bandpass, fwhm=fwhm, output_unit=output_unit
                )
                for idx, row in enumerate(comp_emission):
                    ptsrc_emission[idx] += row

        if fwhm != 0.0:
            # If diffuse emission is non-zero
            print('Smoothing diffuse emission')
            if diffuse_emission.value.any():
                diffuse_emission = hp.smoothing(
                    diffuse_emission, fwhm.to(u.rad).value
                ) * diffuse_emission.unit

        return diffuse_emission + ptsrc_emission


    def disable(self, component: Tuple[str, _Component]) -> None:
        """Disable a component in the model.

        Parameters
        ----------
        component : str, `cosmoglobe.sky._Component`
            The name of a component or the the component class in the 
            model.

        Raises
        ------
        ValueError
            If `component` is not a a `cosmoglobe.sky._Component` or 
            its label.
        KeyError
            If the component is not currently present in the model.
        """

        if isinstance(component, str):
            comp = component
        elif isinstance(component.__class__, _Component):
            comp = component.label
        else:
            raise ValueError(
                'component must be the component label in the model or the '
                'component object'
            )
        try:
            self.disabled_components[comp] = self.components[comp]
        except KeyError:
            raise KeyError(f'{comp} is not enabled')
        del self.components[comp]


    def enable(self, component: Tuple[str, _Component]) -> None:
        """enable a disabled component.

        Parameters
        ----------
        component : str, `cosmoglobe.sky.Component`
            The name of a component or the the component class in the model.
        
        Raises
        ------
        ValueError
            If `component` is not a a `cosmoglobe.sky._Component` or 
            its label.
        KeyError
            If the component is not currently disabled in the model.
        """

        if isinstance(component, str):
            comp = component
        elif isinstance(component.__class__, _Component):
            comp = component.label
        else:
            raise ValueError(
                'component must be the component label in the model or the '
                'component object'
            )

        try:
            self.components[comp] = self.disabled_components[comp]
        except KeyError:
            raise KeyError(f'{comp} is not disabled')
        del self.disabled_components[comp]


    def _insert_component(self, component):
        """Insert a new component to the model.

        Parameters
        ----------
        component : `cosmoglobe.sky.Component`:
            Sky component to be added to the model. Must be a subclass of 
            `cosmoglobe.sky.Component`.

        """
        self._add_component(component)


    def _remove_component(self, name):
        """Removes a component from the model.

        Parameters
        ----------
        name : str
            Component attribute name.
        """

        del self[name]


    def to_nside(self, new_nside: int) -> None:
        """ud_grades all maps in the component to a new nside.

        Parameters
        ----------
        new_nside : int
            Healpix map resolution parameter.

        Raises
        ------
        ValueError
            If NSIDE is not a power of 2.
        """

        if new_nside == self.nside:
            print(f'Model is already at nside {new_nside}')
            return
        if not hp.isnsideok(new_nside, nest=True):
            raise ValueError(f'nside: {new_nside} is not valid.')
        
        self.nside = new_nside
        for comp in self:
            comp.to_nside(new_nside)


    @property
    def is_polarized(self):
        """Returns True if model includes a polarized component and False 
        otherwise.
        """
        for comp in self:
            if comp.is_polarized:
                return True
        return False


    def __iter__(self):
        return iter(self.components.values())


    def __len__(self):
        return len(self.components)


    def __delitem__(self, name):
        if name not in self.components:
            raise KeyError(f'component {name} does not exist')
        delattr(self, name)
        del self.components[name]


    def __repr__(self):
        reprs = []
        for key, component in self.components.items():
            component_repr = repr(component) + '\n'
            reprs.append(f'({key}): {component_repr}')

        main_repr = f'Model('
        main_repr += f'\n  nside: {self.nside}'
        main_repr += '\n  components( '
        main_repr += '\n    ' + '    '.join(reprs)
        main_repr += f'  )'
        main_repr += f'\n)'

        return main_repr