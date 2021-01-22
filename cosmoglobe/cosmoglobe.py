from .models.skycomponent import SkyComponent
from .models.synch import Synchrotron, PowerLaw
from .models.ff import FreeFree
from .models.dust import ModifiedBlackbody
from .models.cmb import CMB
from .models.ame import AME

from .tools.chain import Chain

#list of all foreground components and labels
components = {comp.__name__.lower():comp for comp in SkyComponent.__subclasses__()}
component_labels = {comp.comp_label:comp for comp in SkyComponent.__subclasses__()}


class Cosmoglobe:
    """Cosmoglobe sky model. 

    Provides methods and utilities to process, analyze, and make simulations
    from Commander outputs.
    
    """

    def __init__(self, data):
        """
        Initializes the Cosmoglobe sky model.

        Parameters
        ----------
        data : str
            Path to Commander chain directory or chain h5 file. Defaults to 
            standard Cosmoglobe data directory.

        """
        self.datapath = data
        self.chain = Chain(data)


    def model(self, component_name, **kwargs):
        """
        Initializes a sky component.

        Parameters
        ----------
        component_name : str
            Model name. Name must be contained in the list of available
            cosmoglobe sky models (components or component_labels).

        """
        if component_name.lower() in components:
            component =  components[component_name.lower()]

        elif component_name.lower() in component_labels:
            component = component_labels[component_name.lower()]

        else:
            raise ValueError(
                f"'{component_name}' is not a valid component. Please select "
                f"between the following components:{self.chain.components}"
            )
        models = {model.model_label:model for model in component.__subclasses__()}
        model = models[self.chain.model_params[component.comp_label]['type']]

        return model(self.chain, **kwargs)


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
