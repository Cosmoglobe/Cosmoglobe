from .components import Component

import astropy.units as u
import numpy as np


class Model:
    """A sky model.

    This class acts as a container for the various components making up the 
    sky model. It provides methods that operate on all components.

    TODO: Should all components be smoothed to the same beam before any 
    evaluation can be done? 

    Args:
    -----
    components : list
        A list of cosmoglobe.Component objects that will be added to the model.

    """
    _components = {}
    def __init__(self, components, nside=None, fwhm=None):
        self.nside = nside
        self.fwhm = fwhm
        for component in components:
            self._add_component(component)


    def _add_component(self, component):
        """Adds a component to the current model"""
        if not isinstance(component, Component):
            raise TypeError(f'{component} is not a subclass of Component')

        name = component.comp_name
        if name in self._components:
            raise KeyError(f'component {name} already exists in model')
        if component.amp.nside != self.nside:
            raise ValueError(
                f'component {name!r} has a reference map at NSIDE'
                f'{component.amp.nside}, but model NSIDE is set to {self.nside}'
            )

        self._components[name] = component
        setattr(self, name, component)


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, freq, bandpass=None, output_unit=None):
        """Returns the full model sky emission at an arbitrary frequency.

        TODO: add possibility to choose output_unit.

        Args:
        -----
        freq : astropy.units.quantity.Quantity
            A frequency, or list of frequencies for which to evaluate the 
            component emission.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile corresponding to the frequency list. If None, a 
            delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.Unit
            The desired output units of the emission. Must be signal units, e.g 
            Jy/sr or K.
            Default : None


        Returns
        -------
        astropy.units.quantity.Quantity
            Model emission at the given frequency.

        """
        return sum([comp.get_emission(freq, bandpass, output_unit) for comp in self])
                

    def __iter__(self):
        return iter(self._components.values())


    def __len__(self):
        return len(self._components)


    def __setitem__(self, name, component):
        self._add_component(name, component)


    def __delitem__(self, name):
        if name not in self._components:
            raise KeyError(f'compmonent {name} doesnt exists')

        del self._components[name]
        delattr(self, name)


    def __repr__(self):
        reprs = []
        for key, component in self._components.items():
            component_repr = repr(component) + '\n'
            reprs.append(f'({key}): {component_repr}')

        main_repr = 'Model('
        main_repr += '\n ' + ' '.join(reprs)
        main_repr += ')'

        return main_repr
