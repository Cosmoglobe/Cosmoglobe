from typing import Type
from .components import Component


class Model:

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
        """Adds a component to the currnet model."""
        if not isinstance(component, Component):
            raise TypeError(f'{component} is not a subclass of Component')

        elif name in self._components:
            raise KeyError(f'compmonent {name} already exists')

        self._components[name] = component


    def __iter__(self):
        return iter(self._components)


    def __repr__(self):
        reprs = []
        for key, component in self._components.items():
            component_repr = repr(component) + '\n'
            reprs.append(f'({key}): {component_repr}')

        main_repr = 'Model('
        main_repr += '\n ' + ' '.join(reprs)
        main_repr += ')'

        return main_repr
