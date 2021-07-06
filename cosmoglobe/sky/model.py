from cosmoglobe.sky.components import Component
from cosmoglobe.utils.utils import NsideError, _get_astropy_unit

import astropy.units as u
import healpy as hp
import numpy as np


class Model:
    """A sky model.

    This class acts as a container for the various components making up the
    sky model and provides methods to simulate the sky emission at a given
    frequency or over a bandpass.

    Args:
    -----
    components (list):
        A list of `cosmoglobe.sky.Component` objects that will be added to the
        model.
    nside (int):
        Healpix map resolution. Represents the resolution of the sky model. If
        None, nside will be set to the nside of the first inserted component.
        Default: None

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
        """Simulates the model emission given a single, or, a set of
        frequencies.

        Optionally, a bandpass profile can be given along with the 
        corresponding frequencies resulting in the integrated model emission 
        over the profile.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            A frequency, or list of frequencies for which to evaluate the
            component emission. Must be in units of Hertz.
        bandpass (`astropy.units.Quantity`):
            Bandpass profile corresponding to the frequency list. If None, a
            delta peak in frequency is assumed at the given frequencies.
            Default: None
        output_unit (`astropy.units.Unit`):
            The desired output units of the emission. Must be signal units. 
            Default : uK

        Returns:
        --------
        (`astropy.units.Quantity`)
            Model emission at the given frequency.

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

        Parameters:
        -----------
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

        Parameters:
        -----------
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

        Args:
        -----
        component (a subclass of `cosmoglobe.sky.Component`):
            Sky component to be added to the model. Must be a subclass of 
            `cosmoglobe.sky.Component`.

        """
        self._add_component(component)


    def _remove_component(self, name):
        """Removes a component from the model.

        Args:
        -----
        name (str):
            Component attribute name.

        """
        del self[name]


    def to_nside(self, new_nside):
        """ud_grades all maps in the component to a new nside.

        Args:
        -----
        new_nside (int):
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