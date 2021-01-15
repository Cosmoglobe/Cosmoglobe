import os
import sys
import astropy.units as u 

from .chaintools import get_chainfile, get_component_list, get_params_from_data
from .models.skycomponent import SkyComponent
from .models.synch import Synchrotron, PowerLaw
from .models.ff import FreeFree
from .models.dust import ModifiedBlackbody
from .models.cmb import CMB
from .models.ame import AME

#relative paths to data and models
default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')

#list of all foreground components and labels
components = {comp.__name__.lower():comp for comp in SkyComponent.__subclasses__()}
component_labels = {comp.comp_label:comp for comp in SkyComponent.__subclasses__()}

class Cosmoglobe:
    """Cosmoglobe sky model. 

    Provides methods and utilities to process, analyze, and make simulations
    from Commander outputs.
    
    """

    def __init__(self, data=default_data):
        """
        Initializes the Cosmoglobe sky model.

        Parameters
        ----------
        data : str
            Path to Commander chain directory or chain h5 file. Defaults to 
            standard Cosmoglobe data directory.

        """
        self.datapath = data
        self.data = get_chainfile(data)
        self.loaded_components = get_component_list(self.data)
        self.data_params = get_params_from_data(self.data)


    def model(self, name):
        """
        Initializes a sky component.

        Parameters
        ----------
        name : str
            Model name. Name must be contained in the list of available
            cosmoglobe sky models (components or component_labels).

        """
        if name.lower() in components:
            component =  components[name.lower()]

        elif name.lower() in component_labels:
            component = component_labels[name.lower()]

        else:
            raise ValueError(
                f"'{name}' is not a valid component. Please select between the "
                f"following components:{self.loaded_components}"
            )

        model_params = self.data_params[component.comp_label]
        models = {model.model_label:model for model in component.__subclasses__()}
        model = models[model_params['type']]

        return model(self.data, model_params)


    def __repr__(self):
        """
        Unambigious representation of the Cosmoglobe sky object.

        """
        return f"Cosmoglobe('{self.datapath}')"    
        

    def __str__(self):
        """
        Readable representation of the Cosmoglobe sky object.

        """
        return (
            f"Cosmoglobe sky model generated from "
            f"'{self.datapath}'"
        )