from .models.skycomponent import SkyComponent
from .models.synch import Synchrotron
from .models.ff import FreeFree
from .models.dust import ModifiedBlackbody
from .models.cmb import CMB
from .models.ame import AME

from .tools import chain
from .tools import utils

import numpy as np
import healpy as hp
import astropy.units as u
import pathlib

#list of all foreground components and labels
components = {comp.__name__.lower():comp for comp in SkyComponent.__subclasses__()}
component_labels = {comp.comp_label:comp for comp in SkyComponent.__subclasses__()}


class Cosmoglobe:
    """Cosmoglobe sky model. 

    Provides methods and utilities to analyze, process, and make simulations
    from Commander outputs.
    
    """

    def __init__(self, data, sample='mean', verbose=True):
        """
        Initializes the Cosmoglobe sky object.

        Parameters
        ----------
        data : str
            Path to Commander chain directory or chain h5 file. Defaults to 
            standard Cosmoglobe data directory.

        """
        self.data = data
        self.sample = sample
        self.verbose = verbose

        self.chain = chain.Chain(data, sample)
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
                f'{component_name!r} is not a valid component. Please select ' 
                f'between the following components:\n{*self.chain.components,}'
            )

        models = {model.model_label:model for model in component.__subclasses__()}
        model = models[self.chain.model_params[component.comp_label]['type']]

        if self.verbose:
            print(f'Initializing {model.comp_label}')

        new_model = model(self.chain, **kwargs)
        self.initialized_models.append(new_model)
        
        return new_model


    def spectrum(self, models=None, pol=False, sky_frac=88, min=10, max=1000, num=50):
        """
        Produces a RMS SED for the given models.

        Parameters
        ----------
        models : list, optional
            List of models objects to include in the SED spectrum. Must be a 
            component object, not just the name of a component. Defaults to all
            initialized Cosmoglobe.model objects.
        pol : bool, optional
            If True, the spectrum will be calculated for P = sqrt(Q^2 + U^2). 
            Components that does not include polarization is omitted. Default
            is False.
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
        if self.verbose:
            if pol:
                signal_type = 'P'
            else:
                signal_type = 'I'

            print(
                'Computing SED spectrum with parameters:\n'
                f'  sky_frac: {sky_frac}%\n'
                f'  min frequency: {min} GHz\n'
                f'  max frequency: {max} GHz\n'
                f'  num discrete frequencies: {num}\n'
                f'  signal: {signal_type}'
            )

        if models is None:
            models = self.initialized_models

        if pol:
            for model in models.copy():
                if not model.params['polarization']:
                    print(
                        f'Ignored {model.comp_label} as it does not contain polarization.'
                    )
                    models.remove(model)

            
        mask = utils.create_70GHz_mask(sky_frac)
        freqs = np.logspace(np.log10(min),np.log10(max), num)*u.GHz
        rms_dict = {model.comp_label:[] for model in models}

        for model in models:
            if self.verbose:
                print(f'Calculating RMS for {model.comp_label}')

            if model.params['nside'] > 256:
                model._to_nside(256)
            
            if pol:
                for freq in freqs:
                    amp = model[freq].value
                    Q, U = amp[1], amp[2]
                    P = np.sqrt(Q**2 + U**2)
                    P = hp.ma(P)
                    P.mask = mask
                    rms_dict[model.comp_label].append(np.sqrt(np.mean(P**2)))
            else: 
                for freq in freqs:
                    amp = model[freq].value
                    I = amp[0]
                    I = hp.ma(I)
                    I.mask = mask
                    rms_dict[model.comp_label].append(np.sqrt(np.mean(I**2)))

        return freqs, rms_dict


    def _reduce_chainfile(self, method='mean', save=True, save_filename=None,
                         n_samples=10):
        """
        Reduces a larger chainfile.

        Parameters
        ----------
        chainfile : str, 'pathlib.Path'
            Commander3 chainfile. Must be a hdf5 file.
        method : str, optional
            Method of reduction. Available methods are : mean (draws n_samples 
            random samples, and averages the alm maps). Default is 'mean'.
        save : bool, optional
            If True, outputs a reduced chainfile. If False, returns all parameters
            directly. Default is True.
        save_filename : str, optional
            Filename of output file. If None, and save=True, save_filename is 
            f'reduced_{chainfile.name}'. Default is None
            """
        chain.reduce_chain(self.datapath, 
                           method, 
                           save, 
                           save_filename, 
                           n_samples)


    def __repr__(self):
        if isinstance(self.data, pathlib.PosixPath):
            data = self.data.name
        else:
            data = pathlib.Path(self.data).name

        if self.verbose:
            return (
                f'{self.__class__.__name__}({data!r}, '
                f'sample={self.sample!r})'
            )          
        else:  
            return (
                f'{self.__class__.__name__}({data!r}, '
                f'sample={self.sample!r}, verbose={self.verbose})'
            )
        

    def __str__(self):
        if isinstance(self.data, pathlib.PosixPath):
            data = self.data.name
        else:
            data = pathlib.Path(self.data).name
        return f'Cosmoglobe sky object generated from: {data}'
