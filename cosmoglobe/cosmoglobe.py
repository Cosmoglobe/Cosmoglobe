from .models.skycomponent import SkyComponent
from .models.synch import Synchrotron
from .models.ff import FreeFree
from .models.dust import ModifiedBlackbody
from .models.cmb import CMB
from .models.ame import AME

from .tools.chain import Chain
from .tools import utils

import numpy as np
import healpy as hp
import astropy.units as u

#list of all foreground components and labels
components = {comp.__name__.lower():comp for comp in SkyComponent.__subclasses__()}
component_labels = {comp.comp_label:comp for comp in SkyComponent.__subclasses__()}


class Cosmoglobe:
    """Cosmoglobe sky model. 

    Provides methods and utilities to process, analyze, and make simulations
    from Commander outputs.
    
    """

    def __init__(self, data, verbose=True):
        """
        Initializes the Cosmoglobe sky model.

        Parameters
        ----------
        data : str
            Path to Commander chain directory or chain h5 file. Defaults to 
            standard Cosmoglobe data directory.

        """
        self.datapath = data
        self.verbose = verbose
        self.chain = Chain(data)
        self.initialized_models = []


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

        if self.verbose:
            print(f'Initializing {model.comp_label}')

        new_model = model(self.chain, **kwargs)
        self.initialized_models.append(new_model)
        
        return new_model


    def spectrum(self, models=None, sky_frac=88, min=10, max=1000, num=50):
        """
        Produces a SED spectrum for the given models.

        Parameters
        ----------
        models : list, optional
            List of models objects to include in the SED spectrum. Must be a 
            component object, not just the name of a component. Defaults to all
            initialized Cosmoglobe.model objects.
        sky_frac : int, float, optional
            Fraction of the sky to compute RMS values for. Default is 88%.
        min : int, float, optional
            Minimum value of the frequency range to compute the spectrum over.
        max : int, float, optional
            Maximum value of the frequency range to compute the spectrum over.
        num : int, optional
            Number of discrete frequencies to compute the RMS over. 
            Default is 50.

        Returns
        -------
        frecs: np.ndarray
            Log spaces array of frequencies used to compute the spectrum.
        rms : dict
            Dictionary containing model name and RMS array pairs.

        """
        if models is None:
            models = self.initialized_models

        if self.verbose:
            print(
                'Computing SED spectrum with parameters:\n'
                f'  sky_frac: {sky_frac}%\n'
                f'  min frequency: {min} GHz\n'
                f'  max frequency: {max} GHz\n'
                f'  num discrete frequencies: {num}'
            )

        mask = utils.create_70GHz_mask(sky_frac)
        freqs = np.logspace(np.log10(min),np.log10(max), num)*u.GHz
        rms = [[] for _ in models]
        rms_dict = {}
        for i, model in enumerate(models):
            if self.verbose:
                print(f'Calculating RMS for {model.comp_label}')

            if model.params['nside'] > 256:
                model.to_nside(256)

            for freq in freqs:
                amp = model[freq].value
                amp = hp.ma(amp)
                amp.mask = mask
                rms[i].append(np.sqrt(np.mean(amp**2)))

            rms_dict[model.comp_label] = rms[i]

        return freqs, rms_dict


    def __repr__(self):
        """
        Unambigious representation of the Cosmoglobe sky object.

        """
        return f"cosmoglobe.Cosmoglobe('{self.datapath}')"    
        

    def __str__(self):
        """
        Readable representation of the Cosmoglobe sky object.

        """
        return (
            f"Cosmoglobe sky model - Generated from "
            f"'{self.datapath}'"
        )