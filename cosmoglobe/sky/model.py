from .components import Component

import astropy.units as u
import healpy as hp

class Model:
    """A sky model.

    This class acts as a container for the various components making up the 
    sky model. It provides methods that operate on all components.

    TODO: Should all components be smoothed to the same beam before any 
    evaluation can be done? 

    Args:
    -----
    components (list):
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
        if not issubclass(component.__class__, Component):
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
        freq (astropy.units.quantity.Quantity):
            A frequency, or list of frequencies for which to evaluate the 
            component emission. Must be in units of Hertz.
        bandpass (astropy.units.quantity.Quantity):
            Bandpass profile corresponding to the frequency list. If None, a 
            delta peak in frequency is assumed at the given frequencies. 
            Default: None
        output_unit (astropy.units.Unit):
            The desired output units of the emission. Must be signal units, e.g 
            Jy/sr or K. Default : None


        Returns
        -------
        astropy.units.quantity.Quantity
            Model emission at the given frequency.

        """
        if freq.ndim > 0:
            emissions = []
            for freq in freq:
                emissions.append(sum([comp.get_emission(freq, bandpass, output_unit) 
                                      for comp in self]))
            return emissions

        return sum([comp.get_emission(freq, bandpass, output_unit) for comp in self])


    def insert(self, component):
        """Insert a new component to the model.

        Args:
        -----
        component (a subclass of cosmoglobe.sky.Component):
            Sky component to be added to the model. Must be a subclass of 
            cosmoglobe.sky.Component.

        """
        self._add_component(component)


    def remove(self, name):
        """Removes a component from the model.

        Args:
        -----
        name (str):
            The name of a component present in the model. This is the name in 
            the parenthesis in the model repr.

        """
        del self[name]


    @property
    def component_names(self):
        """Returns a list of the names of the components present in the model"""
        return list(self._components.keys())


    def to_nside(self, new_nside):
        """ud_grades all maps in the component to a new nside.

        Args:
        -----
        new_nside (int):
            Healpix map resolution parameter.

        """
        if new_nside == self.nside:
            return
        if not hp.isnsideok(new_nside, nest=True):
            raise ValueError(f'nside: {new_nside} is not valid.')
        
        for comp in self:
            comp.to_nside(new_nside)



    def __iter__(self):
        return iter(self._components.values())


    def __len__(self):
        return len(self._components)


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