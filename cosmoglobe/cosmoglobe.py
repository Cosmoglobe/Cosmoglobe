import astropy.units as u
import healpy as hp
import numpy as np
import pathlib

from cosmoglobe.models.skycomponent import SkyComponent
from cosmoglobe.models.ame import AME
from cosmoglobe.models.cmb import CMB
from cosmoglobe.models.ff import FreeFree
from cosmoglobe.models.dust import Dust
from cosmoglobe.models.synch import Synchrotron

from cosmoglobe.tools import chain
from cosmoglobe.tools import utils

implemented_comps = {comp.__name__.lower(): comp for comp in SkyComponent.__subclasses__()}
implemented_comp_labels = {comp.comp_label: comp for comp in SkyComponent.__subclasses__()}


class Cosmoglobe:
    """Cosmoglobe sky model

    Provides methods and utilities to analyze, process, and make simulations
    from Commander outputs.
    
    """

    def __init__(self, data, sample='mean', burnin=None, verbose=True):
        """
        Initializes the Cosmoglobe sky object.

        Parameters
        ----------
        data : str
            Path to Commander chain directory or chain h5 file. Defaults to 
            standard Cosmoglobe data directory.
        sample : str, optional
            If sample is 'mean', quantities from the chainfile will be averaged 
            over all samples. Else sample must be a sample number whose 
            quantities will be used.
            Default : 'mean'
        burnin : int, optional
            Discards all samples prior to and including burnin.
            Default : None
        verbose : bool, optional
            If True, provides additional details of the performed computations.
            Default : True

        """
        self.data = data
        self.sample = sample
        self.burnin = burnin
        self.verbose = verbose

        self.chain = chain.Chain(data, sample, burnin)
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
        component_name = component_name.lower()
        if component_name in implemented_comps:
            component = implemented_comps[component_name]

        elif component_name in implemented_comp_labels:
            component = implemented_comp_labels[component_name]

        else:
            raise ValueError(
                f'{component_name!r} is not a valid component. Please select ' 
                f'between the following components:\n{*self.chain.components,}'
            )

        comp_models = {model.model_label: model for model in component.__subclasses__()}
        comp_model = comp_models[self.chain.model_params[component.comp_label]['type']]

        if self.verbose:
            print(f'Initializing {comp_model.comp_label}...')

        model = comp_model(self.chain, **kwargs)
        self.initialized_models.append(model)
        
        return model


    @u.quantity_input(nu=u.Hz)
    def full_sky(self, nu, bandpass=None, output_unit=u.K, models=None):
        """
        Returns the combined emission of a set of models for at a given 
        frequency.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            Frequency at which to evaluate the models.
        models : list
            List of models to compute the emission for.
            Default : None

        Returns
        -------
        full_emission : astropy.units.quantity.Quantity
            Combined emission of all models at the given frequency.

        """
        if models is None:
            if self.initialized_models:
                models = self.initialized_models
            else:
                raise ValueError('No models initialized.')

        full_emission = np.zeros_like(models[0].amp)
        # print(full_emission.unit)
        for model in models:
            if self.verbose:
                if bandpass is None:
                    print(f'Simulating {model.comp_label} at {nu}...')
                else:
                    print(f'Bandpass integrating {model.comp_label}...')
            full_emission += model.get_emission(nu, bandpass, output_unit)


        return full_emission


    @u.quantity_input(start=u.Hz, stop=u.Hz)
    def spectrum(self, models=None, pol=False, sky_frac=88, start=10*u.GHz,
                 stop=1000*u.GHz, num=50):
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
        start : int, float, optional
            Minimum value of the frequency range to compute the spectrum over.
        stop : int, float, optional
            Maximum value of the frequency range to compute the spectrum over.
        num : int, optional
            Number of discrete frequencies to compute the RMS over. 
            Default is 50.

        Returns
        -------
        freqs: np.ndarray
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
                'Making SED spectrum with parameters:\n'
                f'  sky_frac: {sky_frac}%\n'
                f'  start frequency: {start}\n'
                f'  stop frequency: {stop}\n'
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

        start = start.to(u.GHz).value
        stop = stop.to(u.GHz).value
        freqs = np.logspace(np.log10(start), np.log10(stop), num)*u.GHz
        rms_dict = {model.comp_label: [] for model in models}

        for model in models:
            if self.verbose:
                print(f'Computing RMS for {model.comp_label}...')

            if model.params['nside'] != 256:
                model.to_nside(256)

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


    def reduce_chainfile(self, fname=None):
        """
        Reduces a larger chainfile by averaging samples.

        Parameters
        ----------
        fname : str, optional
            Filename of output. If None, fname is f'reduced_{chainfile.name}'.
            Default : None

            """
        chain.reduce_chain(self.data, fname, self.burnin)


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
