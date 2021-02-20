import astropy.units as u
import h5py
import healpy as hp
import pathlib
from numba import njit
import numpy as np
import os

from .data import ModelParameters

ignored_components = ['md', 'bandpass', 'tod', 'gain', 'relquad', 'radio']


class Chain:
    """
    Commander3 chainfile object.

    Operates on chainfiles of the following format:
    -----------------------------------------------
    >>> h5ls chainfile.h5
    sample_1        Group 
    sample_2        Group 
    ...
    sample_n        Group 
    parameters      Group

    >>> h5ls chainfile.h5/sample_1
    component_1     Group 
    component_2     Group 
    ...
    component_n     Group 

    >>> h5ls chainfile.h5/component_1
    param_1         Dataset {scalar, shape(array)}
    param_2         Dataset {scalar, shape(array)}
    ...
    param_n         Dataset {scalar, shape(array)}

    >>> h5ls chainfile.h5/parameters
    component_1     Group 
    component_2     Group 
    ...
    component_n     Group 

    >>> h5ls chainfile.h5/parameters/component_1
    class           Dataset {SCALAR}
    fwhm            Dataset {SCALAR}
    nside           Dataset {SCALAR}
    nu_ref          Dataset {3}
    polarization    Dataset {SCALAR}
    type            Dataset {SCALAR}
    unit            Dataset {SCALAR}

    """

    def __init__(self, chainfile, sample='mean', burn_in=None, verbose=True):
        """
        Initializes the chain object from a give Commander3 chainfile.

        Parameters
        ----------
        chainfile : str, pathlib.Path
            Path to Commander3 chainfile.

        sample : str, optional
            Sample to extract data from. Allowed values are ('sample', 'mean')
            where 'sample' is the name of a sample group in the h5 file. If 
            sample is set to 'mean', the average over all the samples will be
            used. Default is 'mean'.

        """
        if isinstance(sample, str):
            self.sample = sample
        elif isinstance(sample, int):
            self.sample = f'{sample:06d}'
        else:
            raise ValueError(
                "sample must be either a string e.g '000012', or a int e.g 12"
            )
        self.burn_in = burn_in
        validate_chainfile(chainfile, self.sample, self.burn_in)
        self.chainfile = pathlib.Path(chainfile)


        self.samples = self.get_samples()
        self.components = self.get_components()
        self.params = self.get_model_params()

        if verbose:
            print(f'Chain object:')
            print(f'    filename:\t{self.chainfile.name}')
            print(f'    n samples:\t{len(self.samples)}')
            print(f'    burn in:\t{self.burn_in}')
            print(f'    components:\t{*self.components,}\n')
      
    def get_samples(self):
        """
        Returns a list of all samples contained in the chain file.

        """
        with h5py.File(self.chainfile,'r') as f:
            samples = list(f.keys())
            samples.pop(samples.index('parameters'))

        return samples


    def get_components(self):
        """
        Returns a list of all sky components present in a gibbs sample.
        
        """
        with h5py.File(self.chainfile, 'r') as f:
            components = list(set(f['parameters']) - set(ignored_components))
            

        return components


    def get_model_params(self):
        """
        Returns a dictionary containing all model parameters.
        
        """   
        with h5py.File(self.chainfile,'r') as f:
            components = f['parameters']
            model_params = {component: {} for component in components}

            for component in components:
                params = components[component]

                for param in params:
                    value = params[param][()]
                    if isinstance(value, (bytes, np.byte)):
                        model_params[component][param] = params[param].asstr()[()]
                    
                    else:
                        if value.ndim > 0:
                            model_params[component][param] = value
                        else:
                            model_params[component][param] = value.item()

        params = {}
        for component in model_params:
            if model_params[component]['polarization'] == 'True':
                model_params[component]['polarization'] = True            
            elif model_params[component]['polarization'] == 'False':
                model_params[component]['polarization'] = False
                model_params[component]['nu_ref'] = model_params[component]['nu_ref'][0]

            params[component] = ModelParameters(
                type=model_params[component]['type'],
                polarization=model_params[component]['polarization'],
                unit=model_params[component]['unit'],
                nu_ref=model_params[component]['nu_ref']*(1e-9*u.GHz),
                nside=model_params[component]['nside'],
                fwhm=model_params[component]['fwhm']*u.arcmin,
            )

        return params


    def get_alm_list(self, component):
        """
        Returns a list of all available alms for a component in the commander 
        output.

        Parameters
        ----------
        component : str
            Component name.
        
        Returns
        -------
        alms : list
            List containing name of all alm maps for a given component.

        """
        with h5py.File(self.chainfile,'r') as f:
            samples = list(f.keys())
            samples.pop(samples.index('parameters'))
            
            try:
                params = f[samples[0]][component]
            except KeyError:
                raise KeyError(f'"{component}" is not a valid component')

            alms = [param for param in params if 'alm' in param]

            if not alms:
                raise ValueError(f'"{component}" does not include any alms')

        return alms


    def get_item(self, item_name, component, unpack_alms=False, multipoles=None):
        """
        Extracts an item from the chainfile.

        Parameters
        ----------
        item_name : str
            Name of the item to extract.
        component : str
            component name.
        unpack_alms: bool, optional
            If True, item will be asumed to be alms and will be unpacked.
            Default is False.
        multipoles : int, optional
            Specific multipole to unpack. Default is None.

        Returns
        -------
        item : float or numpy.ndarray
            Extracted item.

        """
        if unpack_alms and 'alm' not in item_name:
            raise ValueError(
                'Keyword alm is set to True, but item is not alms'
            )
        
        if multipoles is not None and unpack_alms is False:
            raise ValueError(
                'Keyword unpack_alms must be set to True to get monopoles'
            )

        try:
            self.params[component]
        except KeyError:
            raise KeyError(f'"{component}" is not a valid component')

        with h5py.File(self.chainfile,'r') as f:
            samples = list(f.keys())
            samples.pop(samples.index('parameters'))

            if self.burn_in is not None:
                samples = samples[self.burn_in:]

            if self.sample == 'mean':
                try: 
                    if f[samples[0]][component][item_name].ndim > 0:
                        item = np.zeros_like(f[samples[0]][component][item_name])
                    else:
                        item = 0
                except:
                    raise KeyError(f'{item_name} does not exist in file')
                
                for sample in samples:   
                    item += f[sample][component][item_name][()]

                item /= len(samples)

            else:
                try: 
                    item = f[self.sample][component][item_name][()]
                except:
                    raise KeyError(f'{item_name} does not exist in file')

            if unpack_alms:
                lmax_name = f"{item_name.split('_')[0]}_lmax"
                try:
                    lmax = int(f[samples[0]][component][lmax_name][()])
                except:
                    raise KeyError(f'{lmax_name} does not exist in file')

                if lmax <= 1:
                    pol = False
                else:
                    pol = self.params[component].polarization

                nside = self.params[component].nside
                fwhm = self.params[component].fwhm.to('rad').value

                unpacked_alm = unpack_alms_from_chain(item, lmax)
                items = hp.alm2map(unpacked_alm, 
                                  nside=nside, 
                                  lmax=lmax, 
                                  fwhm=fwhm, 
                                  pol=pol,
                                  verbose=False).astype('float32')

                if multipoles is not None:
                    items = {'amp': items}
                    pole_names = dict(enumerate(['monopole', 'dipole', 'quadrupole']))

                    for multipole in multipoles:
                        unpacked_alm = unpack_alms_multipole_from_chain(item, lmax, multipole)
                        items[pole_names[multipole]] = hp.alm2map(unpacked_alm, 
                                                                  nside=nside, 
                                                                  lmax=lmax, 
                                                                  fwhm=fwhm, 
                                                                  pol=pol,
                                                                  verbose=False).astype('float32') 

                    return items
                return items
        return item[0]


    def __repr__(self):
        return f"Chain(chainfile={self.chainfile.name!r}, sample={self.sample!r}, burn_in={self.burn_in!r})"
    

    def __str__(self):
        return f"Chain object generated from {self.chainfile.name!r}"
    


#Fix to OMP: Error #15 using numba
os.environ['KMP_DUPLICATE_LIB_OK']='True'
@njit
def unpack_alms_multipole_from_chain(data, lmax, multipole):
    """
    Unpacks a single multipole alm from the Commander chain output.

    Parameters
    ----------
    data : HDF5 data
        Chain output hdf5 data.
    lmax : int
        Maximum value for l used in the data.

    Returns
    -------
    alms : numpy.ndarray
        Unpacked version of the Commander alms (2-dimensional array)

    Unpacking algorithm: 
    "https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1
    262/src/tools.py#L9"

    """
    n = len(data)
    n_alms = int(lmax * ((2*lmax + 1 - lmax)/2) + lmax + 1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax+1):
            if l is multipole:
                j_real = l**2 + l
                alms[sigma, i] = complex(data[sigma, j_real], 0.0)
            i += 1
        
        for m in range(1, lmax+1):
            for l in range(m, lmax+1):
                if l is multipole:
                    j_real = l**2 + l + m
                    j_comp = l**2 + l - m
                    alms[sigma, i] = (complex(data[sigma, j_real], 
                                             data[sigma, j_comp],)
                                      /np.sqrt(2.0))
                i += 1
    return alms


@njit
def unpack_alms_from_chain(data, lmax):
    """
    Unpacks alms from the Commander chain output.

    Parameters
    ----------
    data : HDF5 data
        Chain output hdf5 data.
    lmax : int
        Maximum value for l used in the data.

    Returns
    -------
    alms : numpy.ndarray
        Unpacked version of the Commander alms (2-dimensional array)

    Unpacking algorithm: 
    "https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1
    262/src/tools.py#L9"

    """
    n = len(data)
    n_alms = int(lmax * (2*lmax+1 - lmax) / 2 + lmax+1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax+1):
            j_real = l**2 + l
            alms[sigma, i] = complex(data[sigma, j_real], 0.0)
            i += 1

        for m in range(1, lmax+1):
            for l in range(m, lmax+1):
                j_real = l**2 + l + m
                j_comp = l**2 + l - m
                alms[sigma, i] = complex(data[sigma, j_real], 
                                         data[sigma, j_comp],)/ np.sqrt(2.0)
                i += 1

    return alms



def validate_chainfile(chainfile, sample=None, burn_in=None):
    """
    Validates that the given path is of a type and format that matches a
    Commander3 hdf5 output chainfile, and then returns file as a pathlib
    object.
    Parameters
    ----------
    path : str, pathlib.Path
        Path to a Commander chain file or directory containing a HDF5 
        datafile.
    Returns
    -------
    pathlib.Path
        Pathlib Path object to Commander HDF5 chain file.
    """
    try:
        path = pathlib.Path(chainfile)
    except:
        raise TypeError(
            f'Data input to Cosmoglobe() must be str or a os.PathLike '
            'object'
        ) 
    if os.path.isdir(path):
        chainfiles = ([file for file in os.listdir(path) 
                       if file.endswith('.h5') and 'chain' in file])
        if len(chainfiles) == 1:
            path = path / chainfiles[0]
        elif len(chainfiles) > 1:
            raise IOError(
                  f'Data directory contains more than one chain file: '
                  f'{*chainfiles,} \nPlease provide a path including '
                  'the specific chain file you wish to use'
            )
        else:
            raise FileNotFoundError(
                f'{str(path)!r} directory does not contain a valid '
                'chain HDF5 file'
            )
    if os.path.isfile(path):
        if path.name.endswith('.h5'):
            try: 
                with h5py.File(path,'r') as f:
                    try:
                        f['parameters']
                    except KeyError:
                        raise KeyError(
                            'Chainfile does not contain a parameters '
                            'group. Make sure the chainfile was produced '
                            'with a recent version of Commander3'
                        )
                    if len(f) <= 1:
                        raise KeyError(
                            f'Chainfile does not contain any gibbs samples'
                        )
                    elif sample is not None and sample != 'mean':
                        try:
                            f[sample]
                        except KeyError:
                            raise KeyError(
                                'Chainfile does not contain sample '
                                f'{sample!r}'
                            )
                    if burn_in is not None and len(f) < burn_in + 1:
                        raise ValueError(
                            'Burn-in number must be smaller than the '
                            'total sample number'
                        )
            except OSError:
                raise OSError(
                    f'{path.name!r} has an invalid hdf5 format. Make sure '
                    'the file was created with a recent version of '
                    'Commander3.'
                )
        else:
            raise OSError("Input file must be an hdf5 file")
    
    else:
        raise OSError(
        f'No such file or directory: {str(path)!r}'
    )


def reduce_chain(chainfile, fname=None, burn_in=None):
    """
    Reduces a larger chainfile by averaging all, or n randomly selected 
    samples.

    Parameters
    ----------
    fname : str, optional
        Filename of output. If None, fname is f'reduced_{chainfile.name}'.
        Default : None
    burn_in : int, optional
        Discards all samples prior to and including burn_in.
        Default : None
    
        """

    validate_chainfile(chainfile, burn_in=burn_in)
    chainfile = pathlib.Path(chainfile)
    if fname is None:
        fname = f'{chainfile.parent}/reduced_{chainfile.name}'

    with h5py.File(fname, 'w') as reduced_chain:
        chain = h5py.File(chainfile,'r') 
        samples = list(chain.keys())
        parameter_group = samples.pop(samples.index('parameters'))

        chain.copy(parameter_group, reduced_chain)    
        sample_mean = reduced_chain.create_group('sample_mean')

        nsamples = len(samples)
        if burn_in is not None:
            if burn_in >= nsamples:
                raise ValueError(
                    'Burn-in number must be smaller than the '
                    'total sample number'
                )
            else:
                samples = samples[burn_in:]

        maps = {}
        for component in chain[samples[0]]:
            if component not in ignored_components:
                maps[component] = {key: value[()] for key, value in 
                                   chain[samples[0]][component].items()}
                reduced_chain.create_group(f'sample_mean/{component}')

        for sample in samples[1:]:
            components = chain[sample]

            for component in components:
                if component not in ignored_components:
                    parameters = components[component]

                    for parameter in parameters:
                        value = parameters[parameter][()]
                        maps[component][parameter] += value

                        if sample is samples[-1]:
                            if np.issubdtype(value.dtype, np.integer):
                                maps[component][parameter] //= nsamples
                            else:
                                maps[component][parameter] /= nsamples
                            reduced_chain.create_dataset(f'sample_mean/{component}/{parameter}', 
                                                         data=maps[component][parameter], 
                                                         dtype=value.dtype.type)
