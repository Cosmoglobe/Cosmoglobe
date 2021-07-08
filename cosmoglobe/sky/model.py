from cosmoglobe.sky.components import Component
from cosmoglobe.utils.utils import NsideError, _get_astropy_unit

import astropy.units as u
import healpy as hp
import numpy as np


class Model:
    """The Cosmoglobe Sky Model.

    This class acts as a container for the various components making up the
    sky model and provides methods to simulate the sky emission at a given
    frequency or over a bandpass.

    Parameters
    ----------
    components : list
        A list of `cosmoglobe.sky.Component` objects that are added to the
        model.
    nside : int
        Healpix map resolution. Represents the resolution of the sky model. If
        nside is `None`, nside is set to the nside of the first inserted 
        component. Default: ``None``.

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
        Healpix map resolution.
    components : dict
        Dictionary of all sky components included in the model.


    Methods
    -------
    __call__
    disable
    enable
    to_nside

    """

    def __init__(self, components=None, nside=None):
        self.nside = nside
        self.components = {}
        self.disabled_components = {}
        if components:
            for component in components:
                self._add_component(component)


    def _add_component(self, component):
        if not issubclass(component.__class__, Component):
            raise TypeError(
                f'{component} is not a subclass of cosmoglobe.sky.Component'
            )

        name = component.label
        if name in self.components:
            raise KeyError(f'component {name} already exists in model')

        if component.diffuse:
            nside = hp.get_nside(component.amp)
            if nside != self.nside:
                if self.nside is None:
                    self.nside = nside
                else:
                    raise NsideError(
                        f'component {name!r} has a reference map at NSIDE='
                        f'{nside}, but model NSIDE is set to {self.nside}'
                    )
        # Explicitly set nside for non diffuse components since most attributes
        # are not stored in maps           
        else:
            try:
                component._set_nside(self.nside)
            except AttributeError:
                pass
            
        setattr(self, name, component)
        self.components[name] = component


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None), 
                      fwhm=(u.rad, u.deg, u.arcmin))
    def __call__(self, freq, bandpass=None, fwhm=0.0*u.rad, output_unit=u.uK):
        r"""Simulates the model emission given a single or a set of
        frequencies.

        Optionally, a bandpass profile can be given along with the 
        corresponding frequencies.

        .. math::
            
            \begin{aligned}
            \boldsymbol{s}_{\mathrm{RJ}}(\nu) &=\boldsymbol{a}_{\mathrm{CMB}} \frac{x^{2} 
            \mathrm{e}^{x}}{\left(\mathrm{e}^{x}-1\right)^{2}} 
            \frac{\left(\mathrm{e}^{x_{0}}-1\right)^{2}}{x_{0}^{2} 
            \mathrm{e}^{x_{0}}}+\\
            &+\boldsymbol{a}_{\mathrm{s}}\left(\frac{\nu}{\nu_{0, 
            \mathrm{~s}}}\right)^{\beta_{\mathrm{s}}}+\\
            &+\boldsymbol{a}_{\mathrm{ff}} \frac{g_{\mathrm{ff}}
            \left(\nu ; T_{e}\right)}{g_{\mathrm{ff}}\left(\nu_{0, 
            \mathrm{ff}} ; T_{e}\right)}\left(\frac{\nu_{0, 
            \mathrm{ff}}}{\nu}\right)^{2}+\\
            &+\boldsymbol{a}_{\mathrm{sd}}\left(\frac{\nu_{0, 
            \mathrm{sd}}}{\nu}\right)^{2} \frac{s_{0}^{\mathrm{sd}}
            \left(\nu \cdot \frac{\nu_{p}}{30.0 \mathrm{GHz}}\right)}
            {s_{0}^{\mathrm{sd}}\left(\nu_{0, \mathrm{sd}} \cdot 
            \frac{\nu_{p}}{30.0 \mathrm{GHz}}\right)}+\\
            &+\boldsymbol{a}_{\mathrm{d}}\left(\frac{\nu}{\nu_{0, 
            \mathrm{~d}}}\right)^{\beta_{\mathrm{d}}+1} 
            \frac{\mathrm{e}^{h \nu_{0, \mathrm{~d}} 
            / k T_{\mathrm{d}}}-1}{\mathrm{e}^{\mathrm{h} \nu 
            / k T_{\mathrm{d}}}-1}+\\&+\sum_{j=1}^{N_{\mathrm{scc}}} 
            \boldsymbol{a}_{\mathrm{src}}^{j}\left(\frac{\nu}{\nu_{0, 
            \mathrm{src}}}\right)^{\alpha_{j, \mathrm{src}}-2}
            \end{aligned}



        Parameters
        ----------
        freq : `astropy.units.Quantity`
            A frequency, or list of frequencies for which to evaluate the
            sky emission.
        bandpass : `astropy.units.Quantity`
            Bandpass profile corresponding to the frequencies. If None, a
            delta peak in frequency is assumed at the given frequencies.
            Default: None
        output_unit : `astropy.units.Unit`
            The desired output units of the emission. Must be signal units. 
            Default : uK

        Returns
        -------
        astropy.units.Quantity
            Model emission.
        """

        if bandpass is None and freq.ndim > 0:
            return [
                self._get_model_emission(freq, bandpass, fwhm, output_unit)
                for freq in freq
            ]

        return self._get_model_emission(freq, bandpass, fwhm, output_unit)


    def _get_model_emission(self, freq, bandpass, fwhm, output_unit):
        if self.is_polarized:
            shape = (3, hp.nside2npix(self.nside))
        else:
            shape = (1, hp.nside2npix(self.nside))
        diffuse_emission = np.zeros(shape)
        ptsrc_emission = np.zeros(shape)

        unit = _get_astropy_unit(output_unit)
        diffuse_emission = u.Quantity(diffuse_emission, unit=unit)
        ptsrc_emission = u.Quantity(ptsrc_emission, unit=unit)

        for comp in self:
            if comp.diffuse:
                comp_emission = comp(freq, bandpass, output_unit=output_unit)
                for idx, col in enumerate(comp_emission):
                    diffuse_emission[idx] += col
            else:
                comp_emission = comp(freq, bandpass, fwhm=fwhm, output_unit=output_unit)
                for idx, col in enumerate(comp_emission):
                    ptsrc_emission[idx] += col

        if fwhm is not None:
            # If diffuse emission is non-zero
            print('Smoothing diffuse emission')
            if diffuse_emission.value.any():
                diffuse_emission = hp.smoothing(
                    diffuse_emission, fwhm.to(u.rad).value
                ) * diffuse_emission.unit

        return diffuse_emission + ptsrc_emission


    def disable(self, component):
        """Disable a component in the model.

        Parameters
        ----------
        component : str, `cosmoglobe.sky.Component`
            The name of a component or the the component class in the model.
        """

        if isinstance(component, str):
            comp = component
        elif isinstance(component.__class__, Component):
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


    def enable(self, component):
        """enable a disabled component.

        Parameters
        ----------
        component : str, `cosmoglobe.sky.Component`
            The name of a component or the the component class in the model.
        """

        if isinstance(component, str):
            comp = component
        elif isinstance(component.__class__, Component):
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


    def to_nside(self, new_nside):
        """ud_grades all maps in the component to a new nside.

        Parameters
        ----------
        new_nside : int
            Healpix map resolution parameter.
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