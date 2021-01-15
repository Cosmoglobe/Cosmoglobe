"""
Module with functions to read and process Commander3 hdf5 chainfiles.

"""

import h5py
import healpy as hp 
import matplotlib.pyplot as plt
import numba
import numpy as np 
import os 
import pathlib
import astropy.units as u

#Fix to OMP: Error #15 using numba
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_chainfile(data):
    """
    Returns the pathlib.Path object for to the Commander HDF5 chainfile for a
    directory or datafile.

    Parameters
    ----------
    data : str
        Path to a Commander HDF5 datafile or a Commander chain directory
        containing a HDF5 datafile.

    Returns
    -------
    pathlib.Path
        pathlib Path object to Commander HDF5 chain file.

    """
    try:
        path = pathlib.Path(data)
    except:
        raise TypeError(
            f"Data input to Cosmoglobe() must be str or a os.PathLike object."
        )
    if os.path.isfile(data):
        if path.name.endswith('.h5') and 'chain' in path.name:
            return path

    elif os.path.isdir(data):
        for file in os.listdir(data):
            if file.endswith('.h5') and 'chain_test' in file:
                return path.joinpath(file)
        else:
            raise FileNotFoundError(
                f"'{path.name}' directory does not contain a valid "
                "chain HDF5 file."
            )

    else:
        raise FileNotFoundError(
            f"'{path.name}' is not a valid chain HDF5 file or directory path."
        )


def get_params_from_data(data):
    """
    Returns a dictionary containing all model parameters from the commander output.

    Parameters
    ----------
    data : pathlib.Path
        pathlib Path object to Commander HDF5 chain file.

    Returns
    -------
    model_params : dict
        Dictionary containing commander model parameters.

    """   
    model_params = {}
    with h5py.File(data,'r') as f:
        samples = sorted(list(f.keys()))
        if samples[-1] == 'parameters':
            models = f[(f'{samples[-1]}')]
        else:
            raise KeyError(
                f'Chainfile does not contain any model parameters. Make sure the '
                'chainfile was produced with an updated veresion of Commander3.'
        )
        for model in models:
            model_params[model] = {}
            
            params = models[model]
            for param in params:
                if isinstance(models[model][param][()], bytes):
                    model_params[model][param] = models[model][param][()].decode("utf-8")

                else:
                    if len(np.shape(models[model][param][()])) > 0:
                        model_params[model][param] = models[model][param][()]
                    else:
                        model_params[model][param] = models[model][param][()].item()


    for model in model_params:
        model_params[model]['fwhm'] *= u.arcmin
        model_params[model]['nu_ref'] *= u.GHz
        model_params[model]['nu_ref'] *= 1e-9

        if model_params[model]['polarization'] == 'True':
            model_params[model]['polarization'] = True
        elif model_params[model]['polarization'] == 'False':
            model_params[model]['polarization'] = False

    return model_params


def get_item_from_data(data, item, component, sample='000000'):
    """
    Returns a specific item from the commander output.

    Parameters
    ----------
    data : pathlib.Path
        pathlib Path object to Commander HDF5 chain file.
    item : str
        Name of item to be extracted from datafile.
    component : str
        Name of component that contains item.
    sample : str
        Sample number. Default is 000000.

    Returns
    -------
    item : int, float, str, numpy.ndarray
        item extracted from data. Can be any type.

    """  
    with h5py.File(data,'r') as f:
        samples = sorted(list(f.keys()))
        if sample in samples:
            models = f[(f'{sample}')]
        else:
            raise KeyError(
                f"Chainfile does not contain sample '{sample}'."
        )

        components = sorted(list(models.keys()))
        if component in components:
            quantities = models[(f'{component}')]
        else:
            raise KeyError(
                f"Chainfile does not contain data for component: '{component}'."
        )
        
        if item in quantities:
            item = quantities[(f'{item}')]
            if len(np.shape(item)) > 0:
                if np.shape(item[()])[0] == 1:
                    return item[()][0]
                else:
                    return item[()]
            else:
                if np.shape(item[()])[0] == 1:
                    return item[()].item()[0]
                else:
                    return item[()].item()


def get_component_list(data):
    """
    Returns all sky components available in the commander output.

    Parameters
    ----------
    data : pathlib.Path
        pathlib Path object to Commander HDF5 chain file.

    Returns
    -------
    component_list : list of strings
        List containing all avaiable sky components for a given chains file.

    """
    component_list = []
    with h5py.File(data,'r') as f:
        samples = sorted(list(f.keys()))
        components = f[(f'{samples[0]}')]

        for component in components:
            component_list.append(component)

        if not component_list:
            raise ValueError(f'"{data.name}" doest not include any components.')

    return component_list


def get_alm_list(data, component):
    """
    Returns a list of all available alms for a component in the commander 
    output.

    Parameters
    ----------
    data : pathlib.Path
        pathlib Path object to Commander HDF5 chain file.
    component : str
        Foreground name.
    
    Returns
    -------
    alm_list : list of strings
        List containing all avaiable alms for a given component.

    """
    alm_list = []
    with h5py.File(data,'r') as f:
        samples = sorted(list(f.keys()))
        try:
            params = f[(f'{samples[0]}')][component]
        except:
            raise KeyError(f'"{component}" is not a valid component.')

        for param in params:
            if 'alm' in param:
                alm_param = param.split('_')[0]
                alm_list.append(alm_param)
        
        if not alm_list:
            raise ValueError(f'"{component}" doest not include any alms.')

    return alm_list


def get_alms(data, component, nside, param, polarization, fwhm):
    """
    Returns array or arrays containing alms from Commander chain HDF5 file.

    Parameters
    ----------
    data : pathlib.Path
        pathlib Path object to Commander HDF5 chain file.
    component : str
        Foreground name.
    nside: int
        Healpix map resolution.
    param : str, optional
        alm parameter ('amp', 'T', 'beta') to extract from hdf5 file.
        Default is 'amp'.
    lmax: int, optional
        Maximum value for l used in dataset.
        Default is None.
    fwhm : float, scalar, optional
        The fwhm of the Gaussian used to smooth the map (applied on alm)
        [in degrees]
        Default is 0.0.

    Returns
    -------
    alm_map : 'numpy.ndarray' or list of 'numpy.ndarray'
        A Healpix map in RING scheme at nside or a list of I,Q,U maps (if
        polarized input)

    """
    with h5py.File(data,'r') as f:
        samples = sorted(list(f.keys()))

        try:
            dataset = f[(f'{samples[0]}')][component]
        except:
            raise KeyError(f'"{component}" is not a valid component.')

        try:
            alms = dataset[f'{param}_alm'][()]
        except:
            raise KeyError(f'"{param}_alm" does not exist in file.')
        
        try:
            lmax = dataset[f'{param}_lmax'][()]
        except:
            raise KeyError(f'"{param}_lmax" does not exist in file.')

        if lmax <= 1:
            polarization = False

        alms_unpacked = unpack_alms_from_chain(alms, lmax)
        validate_nside(nside)
        alms_map = hp.alm2map(alms_unpacked, nside, lmax=lmax, mmax=lmax,
                              fwhm=fwhm.to('rad').value, pol=polarization,
                              pixwin=True)

    return alms_map


@numba.njit(cache=True, fastmath=True)
def unpack_alms_from_chain(dataset, lmax):
    """
    Unpacks alms from the Commander chain output.

    Parameters
    ----------
    dataset : HDF5 dataset
        Chain output hdf5 dataset.
    lmax : int
        Maximum value for l used in the dataset.

    Returns
    -------
    alms : 'numpy.ndarray'
        Unpacked version of the Commander alms (2-dimensional array)

    Unpacking algorithm: 
    "https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1
    262/src/tools.py#L9"

    """
    n = len(dataset)
    n_alms = int(lmax * (2*lmax+1 - lmax) / 2 + lmax+1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax+1):
            j_real = l**2 + l
            alms[sigma, i] = complex(dataset[sigma, j_real], 0.0)
            i += 1
        
        for m in range(1, lmax+1):
            for l in range(m, lmax+1):
                j_real = l**2 + l + m
                j_comp = l**2 + l - m
                alms[sigma, i] = complex(dataset[sigma, j_real], 
                                         dataset[sigma, j_comp],)/ np.sqrt(2.0)
                i += 1

    return alms


def validate_nside(nside):
    """
    Raises a ValueError if the given nside is invalid.

    Parameters
    ----------
    nside : int
        Healpix map resolution parameter.

    """
    if not isinstance(nside, int) or nside < 0:
        raise ValueError('nside must be a positive integer.')

    if not np.log2(nside).is_integer():
        raise ValueError(f"'{nside}' is not a valid nside value.")

