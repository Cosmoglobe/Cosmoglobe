from .components import Component

import astropy.units as u
import numpy as np


class Model:
    """A sky model.

    This class acts as a container for the various components making up the 
    sky model. It provides methods that operate on all components.

    Args:
    -----
    components : list
        A list of cosmoglobe.Component objects that will be added to the model.

    """
    _components = {}
    def __init__(self, components):
        for component in components:
            self._add_component(component)


    def _add_component(self, component):
        """Adds a component to the current model"""
        if not isinstance(component, Component):
            raise TypeError(f'{component} is not a subclass of Component')

        name = component.comp_name
        if name in self._components:
            raise KeyError(f'component {name} already exists in model')

        self._components[name] = component
        setattr(self, name, component)


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=None):
        """Returns the full model sky emission at an arbitrary frequency nu.

        Args:
        -----
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the 
            component emission.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile corresponding to the frequency array, nu. 
            If None, a delta peak in frequency is assumed.
            Default : None

        Returns
        -------
        astropy.units.quantity.Quantity
            Model emission at the given frequency in units of output_unit.

        """
        return sum([comp.get_emission(nu, bandpass, output_unit) for comp in self])
                

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
