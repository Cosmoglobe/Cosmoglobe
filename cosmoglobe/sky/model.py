from cosmoglobe.utils import utils
from cosmoglobe.utils.utils import CompState
from cosmoglobe.sky.base import (
    _Component, 
    _DiffuseComponent, 
    _PointSourceComponent,
)

from typing import List
import astropy.units as u
import healpy as hp
import numpy as np


class Model:
    r"""Sky model object representing the Cosmoglobe Sky Model.

    This class acts as a container for the various components making up 
    the Cosmoglobe Sky Model, and provides methods to simulate the sky. 
    The primary use case of this class is to call its ``__call__`` 
    function, which simulates the sky at a single frequency :math:`\nu`, 
    or integrated over a bandpass :math:`\tau`.
    
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

    Methods
    -------
    __call__
    disable
    enable

    See Also
    --------
    __call__ : 
        Simulates the sky at a given frequency :math:`\nu` or over a 
        bandpass :math:`\tau` given the Cosmoglobe Sky Model.

    Notes
    -----
    The model object is self is an iterable:
    the model components as following:

    >>> model = skymodel(nside=256)
    >>> for component in model:
            print(component)
    AME(nu_p)
    CMB()
    Dust(beta, T)
    FreeFree(Te)
    Radio(specind)
    Synchrotron(beta)
    """

    nside: int = None 
    components: List[_Component] = None, 

    def __init__(self, nside=None, components=None):
        """Initializing a sky model. 

        Parameters
        ----------
        nside : int, optional
            Healpix resolution of the maps in sky model (the default is 
            None, in which the model automatically detects the nside from
            the components).
        components : list, optional
            A list of `cosmoglobe.sky.base._Component` objects that 
            constitutes the sky model (by default this is None and the 
            components are iteratively added as they are read from a 
            commander3 chain).
        """

        self.nside = nside
        self._components = {}

        if components is not None:
            for component in components:
                self._add_component_to_model(component)

    def _add_component_to_model(self, component):
        name = component.label
        if name in self._components:
            raise KeyError(
                f'component {name} is already a part of the model'
            )

        setattr(self, name, component)
        self._components[name] = [component, CompState.ENABLED]

        if self.nside is None:
            self.nside = hp.get_nside(component.amp)

    @u.quantity_input(
        freq=u.Hz, 
        bandpass=(u.Jy/u.sr, u.K, None), 
        fwhm=(u.rad, u.deg, u.arcmin)
    )
    def __call__(
        self, freqs, bandpass=None, fwhm=0.0 * u.rad, output_unit=u.uK
    ):
        r"""Simulates the full model sky emission. 

        This method is the main use case of the sky model object. It 
        simulates the full model sky emission (sum of all component 
        emission) a single frequency :math:`\nu` or integrated over a 
        bandpass :math:`\tau`.

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
        output_unit : `astropy.units.UnitBase`, optional
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
        Simulated full sky emission at :math:`50\; \mathrm{GHz}`:

        >>> from cosmoglobe import skymodel
        >>> import astropy.units as u
        >>> model = skymodel(nside=256) 
        >>> model(50*u.GHz)[0]
        [ 2234.74893115  2291.99921295  2323.98779311 ... -2320.74732945
         -2271.54465982 -2292.22248419] uK

        Simulated full sky emission at :math:`500\; \mathrm{GHz}` 
        smoothed with a :math:`50\; '` Gaussian beam, outputed in units of 
        :math:`\mathrm{MJy} / \mathrm{sr}`:

        >>> model(500*u.GHz, fwhm=50*u.arcmin, output_unit='MJy/sr')[0]
        [ 0.267749    0.26991688  0.28053964 ... -0.15846278 -0.15269807
         -0.14408377] MJy / sr
        """

        shape = (3, hp.nside2npix(self.nside))
        diffuse_emission = np.zeros(shape)
        point_source_emission = np.zeros(shape)

        # The output unit may be a string denoting for instance K_CMB, which
        # is critical information for the following routines. However, we 
        # need to initialize the emission arrays with astropy units.
        _output_unit = utils.str_to_astropy_unit(output_unit)
        diffuse_emission = u.Quantity(diffuse_emission, unit=_output_unit)
        point_source_emission = u.Quantity(
            point_source_emission, unit=_output_unit
        )

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
                    point_source_emission[idx] += row

        if fwhm != 0.0:
            # If diffuse emission is non-zero
            print('Smoothing diffuse emission')
            if diffuse_emission.value.any():
                diffuse_emission = hp.smoothing(
                    diffuse_emission, fwhm.to(u.rad).value
                ) * diffuse_emission.unit

        return diffuse_emission + point_source_emission

    def disable(self, component):
        """Disable a component in the model.

        Parameters
        ----------
        component : str, `cosmoglobe.sky.base._Component`
            The name of a component or the the component class in the 
            model.

        Raises
        ------
        KeyError
            If the component is not currently enabled in the model.
        """

        try:
            comp = component.label
        except AttributeError:
            comp = component

        if self._components[comp][1] is CompState.ENABLED:
            self._components[comp][1] = CompState.DISABLED
        else:
            raise KeyError(f'{comp} is already disabled')

    def enable(self, component):
        """Enable a disabled component.

        Parameters
        ----------
        component : str, `cosmoglobe.sky.Component`
            The name of a component or the the component class in the model.
        
        Raises
        ------
        KeyError
            If the component is not currently disabled in the model.
        """

        try:
            comp = component.label
        except AttributeError:
            comp = component

        if self._components[comp][1] is CompState.DISABLED:
            self._components[comp][1] = CompState.ENABLED
        else:
            raise KeyError(f'{comp} is already enabled')

    def __iter__(self):
        """Returns iterable of all enabled components in the model."""
        
        components = [
            comp[0] for comp in self._components.values()
            if comp[1] is CompState.ENABLED
        ]
        return iter(components)

    def __repr__(self):        
        """Representation of the Model and all enabled components."""

        reprs = []
        for key, component in self._components.items():
            if component[1] is CompState.ENABLED:
                component_repr = repr(component[0]) + '\n'
                reprs.append(f'({key}): {component_repr}')

        main_repr = f'Model('
        main_repr += f'\n  nside: {self.nside}'
        main_repr += '\n  components( '
        main_repr += '\n    ' + '    '.join(reprs)
        main_repr += f'  )'
        main_repr += f'\n)'

        return main_repr