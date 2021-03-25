from typing import Type
from .components import Component


class Model:
    """A model of the sky.

    This class acts as a container for the various components making up the sky 
    model.

    Args:
    -----
    components : list of tuples
        A list of tuples, containing the names and component objects that will 
        be included in the model.

    """
    _components = {}
    def __init__(self, components):
        for component in components:
            if isinstance(component, (tuple, list)):
                name, component = component
            else:
                name = component.__class__.__name__

            self._add_component(name, component)
            setattr(self, name, component)


    def _add_component(self, name, component):
        """Adds a component to the current model"""
        if not isinstance(component, Component):
            raise TypeError(f'{component} is not a subclass of Component')

        elif name in self._components:
            raise KeyError(f'compmonent {name} already exists')

        self._components[name] = component


    def __iter__(self):
        return iter(self._components.values())


    def __len__(self):
        return len(self._components)


    def __setitem__(self, name, component):
        self._add_component(name, component)
        return setattr(self, name, component)

    def __delitem__(self, name):
        if name not in self._components:
            raise KeyError(f'compmonent {name} doesnt exists')

        del self._components[name]


    def __repr__(self):
        reprs = []
        for key, component in self._components.items():
            component_repr = repr(component) + '\n'
            reprs.append(f'({key}): {component_repr}')

        main_repr = 'Model('
        main_repr += '\n ' + ' '.join(reprs)
        main_repr += ')'

        return main_repr
