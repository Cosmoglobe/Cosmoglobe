import pathlib
import astropy.units as u 
import healpy as hp 
import numpy as np

from cosmoglobe.sky.models.skycomponent import SkyComponent
from cosmoglobe.sky.models.ame import AME
from cosmoglobe.sky.models.cmb import CMB
from cosmoglobe.sky.models.ff import FreeFree
from cosmoglobe.sky.models.dust import Dust
from cosmoglobe.sky.models.synch import Synchrotron

from cosmoglobe.tools import chain
from cosmoglobe.tools import utils
from cosmoglobe.tools.data import (
    get_params_from_config,
    unpack_config, 
)


class SkyModel:
    """Cosmoglobe sky model.

    Provides methods and utilities to analyze and make simulations of 
    cosmological sky models.
    
    """
    def __init__(self, datafile, components=None, nside=None, fwhm=None, 
                 sample='mean', burn_in=None, verbose=True):

        """
        Initializes Sky Model components, either from a Commander3 chain file
        file, or from a config json file.

        Parameters
        ----------
        datafile : str
            Path to Commander chain directory, chain h5 file or a config file. 
            (will default to standard Cosmoglobe data in the future).
        components : list
            List of sky component names, i.e 'dust', 'synch', etc, that will 
            be initialized. If None, all components included in chain or 
            config file will be included.
            Default : None
        nside: int
            Healpix map resolution. Sets model maps to nside.
            Default : None
        fwhm : float
            The fwhm of the Gaussian used to smooth the map. Sets model maps 
            to fwhm.
            Default : None
        sample : str, optional
            If sample is 'mean', quantities from the chainfile will be averaged 
            over all samples. Else sample must be a sample number whose 
            quantities will be used.
            Default : 'mean'
        burn_in : int, optional
            Discards all samples prior to and including burn_in.
            Default : None
        verbose : bool, optional
            If True, provides additional details of the performed computations.
            Default : True
        
        Raises
        ------
        ValueError

        """
        try:
            self.datafile = pathlib.Path(datafile)
        except TypeError:
            raise TypeError(
                'Datafile type must be a string, or a pathlib.Path object, '
                f'not of type {type(datafile)}'
            )        
        self.nside = nside
        self.fwhm = fwhm
        self.verbose = verbose
        self.components = []

        if self.datafile.suffix == '.h5':
            self.data = chain.Chain(chainfile=datafile, 
                                    sample=sample, 
                                    burn_in=burn_in, 
                                    verbose=verbose)
            self.params = self.data.params
            if components is None:
                components = self.data.components
            else:
                if not all(comp.lower() in self.data.components 
                           for comp in components):
                    raise ValueError(
                        'Component names not matching with chainfile names. '
                        f'Included comps in chainfile: {*self.data.components,}'
                    )
                
        elif self.datafile.suffix == '.json':
            self.data = unpack_config(datafile)
            if components is None:
                components = self.data.keys()
            config_params = get_params_from_config(self.data)
            self.params = {comp : params for comp, params 
                           in config_params.items() if comp in components}
        else:
            raise TypeError(
                f'datafile: {self.datafile.name!r} must be either a '
                'Commander3 .h5 chainfile, or a .json configuration file.'
            )

        skycomponents = self._get_skycomponents(components)
        for component in skycomponents:
            if not self.verbose:
                print(
                    f"Initializing {component.comp_label}...",
                )
            else:
                print(
                    "Initializing sky component:\n",
                    f"    label:\t{component.comp_label}\n",
                    f"    type:\t{self.params[component.comp_label].type}\n",
                    f"    polarized:\t{self.params[component.comp_label].polarization}\n",
                    f"    unit:\t{self.params[component.comp_label].unit}\n",
                    f"    ref freq:\t{self.params[component.comp_label].nu_ref}\n",
                    f"    nside:\t{self.params[component.comp_label].nside}\n",
                    f"    fwhm:\t{self.params[component.comp_label].fwhm}\n",
                    "...\n"
                )
            model = component(self.data, nside=nside, fwhm=fwhm)
            self.components.append(model)

            setattr(self, model.comp_label, model)
    

    def _get_skycomponents(self, components):
        """
        Returns a list of SkyComponent subclasses matching input components.
        TODO: Add support for specific requested models. Perhaps in the form 
        of tuple inputs, i.e if components=['ff', ('ame', 'spindust2')]...
        """
        skycomponents = []
        for comp in components:
            for class_ in SkyComponent.__subclasses__():
                if comp.lower() in (class_.comp_label, class_.__name__.lower()):
                    class_models = {model.model_label: model 
                                   for model in class_.__subclasses__()}
                    skycomponents.append(
                        class_models[self.params[class_.comp_label].type]
                    )
                    break

            else:
                raise ValueError(f'{comp=} is not implemented')

        return skycomponents


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=u.K):
        """
        Returns the sky model emission at an arbitrary frequency nu or for a 
        bandpass profile, in units of K_RJ.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the model.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile in units of (k_RJ, Jy/sr) corresponding to 
            frequency array nu. If None, a delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.quantity.Quantity or str
            Desired unit for the output map. Must be a valid astropy.unit or 
            one of the two following strings ('K_CMB', 'K_RJ').
            Default : None

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """

        return np.sum([comp.get_emission(nu, bandpass, output_unit) 
                      for comp in self.components], axis=0)


    @u.quantity_input(start=u.Hz, stop=u.Hz)
    def get_spectrum(self, components=None, pol=False, sky_frac=88, start=10*u.GHz,
                     stop=1000*u.GHz, num=50):
        """
        Produces a RMS SED spectrum for all included sky components.

        Parameters
        ----------
        component : list, optional
            List of sky components to include. Must be a component object, 
            not just the name of a component. If None, then all initialized 
            SkyComponent objects are used.
            Default : None
        pol : bool, optional
            If True, the spectrum will be calculated for P = sqrt(Q**2 + U**2). 
            Components that does not include polarization are omitted. 
            Default : False.
        sky_frac : int, float, optional
            Fraction of the sky to compute RMS values for. 
            Default : 88
        start : int, float, optional
            Minimum value of the frequency range to compute the spectrum over.
            Default : 10 GHz
        stop : int, float, optional
            Maximum value of the frequency range to compute the spectrum over.
            Default : 1000 GHz
        num : int, optional
            Number of discrete frequencies to compute the RMS over. 
            Default : 50

        Returns
        -------
        freqs: np.ndarray
            Log spaces array of frequencies used to compute the spectrum.
        rms : dict
            Dictionary containing model name and RMS array pairs.

        """
        print(
            'Computing SED spectrum...'
        )
        if self.verbose:
            if pol:
                signal_type = 'P'
            else:
                signal_type = 'I'
            print(
                f'    signal :\t\t{signal_type}\n'
                f'    sky fraction:\t{sky_frac}%\n'
                f'    start :\t\t{start}\n'
                f'    stop :\t\t{stop}\n'
                f'    n : \t\t{num}\n',
            )

        if components is None:
            components = self.components

        if pol:
            for model in components.copy():
                if not model.params.polarization:
                    print(
                        f'Ignored {model.comp_label} as it does not contain polarization.'
                    )
                    components.remove(model)

            
        mask = utils.create_70GHz_mask(sky_frac)

        start = start.to(u.GHz).value
        stop = stop.to(u.GHz).value
        freqs = np.logspace(np.log10(start), np.log10(stop), num)*u.GHz
        rms_dict = {component.comp_label: [] for component in components}

        if self.verbose:
            print(f'    Computing RMS values:')
        for model in components:
            if self.verbose:
                print(f'        {model.comp_label}...')

            if model.params.nside != 256:
                model.to_nside(256)

            for freq in freqs:
                I, Q, U = model.get_emission(freq).value
                if pol:
                    P = np.sqrt(Q**2 + U**2)
                    P = hp.ma(P)
                    P.mask = mask
                    rms_dict[model.comp_label].append(np.sqrt(np.mean(P**2))) 
                else:
                    I = hp.ma(I)
                    I.mask = mask
                    rms_dict[model.comp_label].append(np.sqrt(np.mean(I**2)))

        return freqs, rms_dict


def reduce_chain(chainfile, fname=None, burn_in=None):
    """
    Output a reduced version of the input chainfile containing only a 
    single averaged sample group and a parameter group.
    
    """
    print(f'Reducing {chainfile=}...')
    chain.reduce_chain(chainfile, fname=fname, burn_in=burn_in)
