import inspect
import pathlib
import sys

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from numba import njit
from tqdm import tqdm

from cosmoglobe.sky import components
from cosmoglobe.sky import Model
from cosmoglobe.utils import utils

# Model parameter group name as implemented in commander
param_group = 'parameters'
# These will be dropped from component lists
_ignored_comps = ['md', 'relquad']

#Current Cosmoglobe Sky Model as of BP9
COSMOGLOBE_COMPS = dict(
    ame=components.AME,
    cmb=components.CMB,
    dust=components.Dust,
    ff=components.FreeFree,
    radio=components.Radio,
    synch=components.Synchrotron,
)


def model_from_chain(file, nside=None, samples='all', burn_in=None, comps=None):
    """Returns a sky model from a commander3 chainfile.
    
    A cosmoglobe.sky.Model is initialized that represents the sky model used in 
    the given Commander run.

    Args:
    -----
    file (str):
        Path to commander3 hdf5 chainfile.
    nside (int):
        Healpix resolution parameter.
    sample (str, int, list):
        If sample is None, then the model will be initialized from sample 
        averaged maps. If sample is a string (or an int), the model will be 
        initialized from that specific sample. If sample is a list, then the 
        model will be initialized from an average over the samples in the list.
        Default: None
    burn_in (str, int):
        The sample number as a str or int where the chainfile is assumed to 
        have sufficently burned in. All samples before the burn_in are ignored 
        in the averaging process.
    comps (dict):
        Dictionary of which classes to use for each component. The keys must
        the comp group names in the chain file. If comps is None, a default set
        of components will be selected. Default: None

    Returns:
    --------
    (cosmoglobe.sky.Model):
        A sky model representing the results of a commander3 run.

    """
    model = Model(nside=nside)

    default_comps = COSMOGLOBE_COMPS

    if comps is None:
        comps = default_comps
    else:
        if not isinstance(comps, list):
            comps = [comps]
        comps = {comp:default_comps[comp] for comp in comps}

    if not comps:
        raise ValueError('No comps selected')

    chain_components = _get_components(file)
    component_list = [comp for comp in comps if comp in chain_components]

    if samples == 'all':
        samples = _get_samples(file)
    elif samples == -1:
        samples = _get_samples(file)[-1]
    elif isinstance(samples, int):
        samples = _int_to_sample(samples)
    if burn_in is not None:
        if len(samples) > burn_in:
            samples = samples[burn_in:]
        else:
            raise ValueError('burn_in sample is out of range')

    print('Loading components from chain')
    with tqdm(total=len(component_list), file=sys.stdout) as pbar:
        padding = len(max(component_list, key=len))
        for comp in component_list:
            pbar.set_description(f'{comp:<{padding}}')
            comp = comp_from_chain(file, comp, comps[comp], 
                                     nside, samples)                                           
            model._add_component_to_model(comp)
            pbar.update(1)

        pbar.set_description('Done')

    return model


def comp_from_chain(file, component, component_class, model_nside, samples):
    """Returns a sky component from a commander3 chainfile.
    
    A sky component that subclasses cosmoglobe.sky.Component is initialized 
    from a given commander run.

    Args:
    -----
    file (str):
        Path to commander3 hdf5 chainfile.
    component (str):
        Name of a sky component. Must match the hdf5 component group name.
    sample (str, int, list):
        If sample is 'mean', then the model will be initialized from sample 
        averaged maps. If sample is a string (or an int), the model will be 
        initialized from that specific sample. If sample is a list, then the 
        model will be initialized from an average over the samples in the list.
        Default: 'mean'
    burn_in (str, int):
        The sample number as a str or int where the chainfile is assumed to 
        have sufficently burned in. All samples before the burn_in are ignored 
        in the averaging process.

    Returns:
    --------
    (sub class of cosmoglobe.sky.Component):
        A sky component initialized from a chain.

    """
    # Getting component parameters from chain
    parameters = _get_component_params(file, component)
    freq_ref = (parameters['nu_ref']*u.Hz).to(u.GHz)
    fwhm_ref = (parameters['fwhm']*u.arcmin).to(u.rad)
    nside = parameters['nside']

    if parameters['polarization'] == 'True':
        comp_is_polarized = True
    else:
        comp_is_polarized = False

    # Commander outputs units in uK_RJ for all comps except for CMB which is in
    # K_CMB. This is manually handeled in the CMB component. NB! If this 
    # changes in future Commander versions, this part needs to be updated.
    amp_unit = u.uK

    # Getting arguments required to initialize component
    args_list = _get_comp_args(component_class) 

    args = {}
    if isinstance(samples, list) and len(samples) > 1:
        get_items = _get_averaged_items
    else:
        get_items = _get_items

    # Find which args are alms and which are precomputed maps
    alm_names = []
    map_names = []
    other_items_names = []
    for arg in args_list:
        if arg != 'freq_ref':
            if _item_alm_exists(file, component, arg):
                alm_names.append(arg)
            elif _item_map_exists(file,component, arg):
                map_names.append(arg)
            elif _item_exists(file, component, arg):
                other_items_names.append(arg)
            elif arg == 'nside':
                args['nside'] = nside
            else:
                raise KeyError(f'item {arg} is not present in the chain')


    other_items_ = get_items(
        file, samples, component, [item for item in other_items_names]
    )
    other_items = dict(zip(other_items_names, other_items_))
    maps_ = get_items(
        file, samples, component, [f'{map_}_map' for map_ in map_names]
    )

    maps = dict(zip(map_names, maps_))
    if maps:
        if model_nside is not None and nside != model_nside:
            maps = {key:hp.ud_grade(value, model_nside) 
                    if isinstance(value, np.ndarray) 
                    else value for key, value in maps.items()}
    args.update(maps)
    args.update(other_items)
    if model_nside is None:
        model_nside = nside

    alms_ = get_items(
        file, samples, component, [f'{alm}_alm' for alm in alm_names]
    )
    alms = dict(zip(alm_names, alms_))
    alms_lmax_ = _get_items(
        file, samples[-1], component, [f'{alm}_lmax' for alm in alm_names]
    )
    alms_lmax = dict(zip(alm_names, [int(lmax) for lmax in alms_lmax_]))

    for key, value in alms.items():
        unpacked_alm = unpack_alms_from_chain(value, alms_lmax[key])
        if key == 'amp' and value.shape[0] == 3:
            pol = True
        else:
            pol = False
                
        alms[key] = hp.alm2map(unpacked_alm, 
                          nside=model_nside, 
                          lmax=alms_lmax[key], 
                          fwhm=fwhm_ref.value,
                          pol=pol,
                          ).astype('float32')

    args.update(alms)
    args['amp'] *= amp_unit
    args = utils.set_spectral_units(args)
    scalars = utils.extract_scalars(args) # dont save scalar maps
    args.update(scalars)
    if 'freq_ref' in args_list:
        if comp_is_polarized:
            freq = u.Quantity(freq_ref[:-1])
        else:
            freq = u.Quantity(freq_ref[0])
        args['freq_ref'] = freq

    if component =='radio':
       args['specind'] = args['specind'][0] # take alpha from chain

    return component_class(**args)


def _get_comp_args(component_class):
    """Returns a list of arguments needed to initialize a component"""
    ignored_args = ['self']
    arguments = inspect.getargspec(component_class.__init__).args
    return [arg for arg in arguments if arg not in ignored_args]


def _get_samples(file):
    """Returns a list of all samples present in a chain file"""
    with h5py.File(file, 'r') as f:
        samples = list(f.keys())

    try:
        samples.remove(param_group)
    except:
        print("Warning: Using an old h5 commander format without param_group")

    return samples


def _sample_to_int(samples, start=0):
    """Converts a sample or a list of samples to integers"""
    if isinstance(samples, (list, tuple)):
        return [int(sample) + start for sample in samples]
    
    return int(samples) + start


def _int_to_sample(samples, start=0):
    """Converts an integer or multiple integers to sample string format"""
    if isinstance(samples, (list, tuple)):
        return [f'{sample + start:06d}' for sample in samples]
    
    return f'{samples + start:06d}'


def _get_components(file, ignore_comps=True):
    """Returns a list of all components present in a chain file"""
    with h5py.File(file, 'r') as f:
        components = list(f[f'{1:06d}'].keys())
    if ignore_comps:    
        return [comp for comp in components if comp not in _ignored_comps]

    return components


def _get_component_params(file, component):
    """Returns a dictionary of the model parameters of a component"""
    return_params = {}
    with h5py.File(file, 'r') as f:
        params = f[param_group][component]
        for param, value in params.items():
            if isinstance(value[()], bytes):
                return_params[param] = value.asstr()[()]
            else:
                return_params[param] = value[()]

        return return_params
    

def _get_items(file, sample, component, items):
    """Return the value of one or many items for a component in the chain file.

    Args:
    -----
    file: str
        Path to h5 file.
    sample : str
        sample name.
    component: str
        Component group name.
    items: str, list, tuple
        Name of item to extract, or a list of names.

    Returns:
    --------
    list
        List of items extracted from the chain file.
    
    """
    with h5py.File(file, 'r') as f:
        items_to_return = []
        try:
            for item in items:
                items_to_return.append(f[sample][component][item][()])

            return items_to_return
        except TypeError:
            return f[sample][component][items][()]


def _get_averaged_items(file, samples, component, items):
    """Return the averaged value of one or many item for a component in the 
    chain file.

    Args:
    -----
    file: str
        Path to h5 file.
    samples : list
        List of samples to average over.
    component:
        Component group.
    items: str, list, tuple
        Name of item to extract, or a list of names. Items must be of types
        compatible with integer division.

    Returns:
    --------
    list
        List of items averaged over samples from the chain file.
    
    """
    if not items:
        return []

    with h5py.File(file, 'r') as f:
        if isinstance(items, (tuple, list)):
            items_to_return = []
            for sample in samples:
                
                for idx, item in enumerate(items):
                    try:
                        items_to_return[idx] += (
                            f[sample][component][item][()]
                        )
                    except IndexError:
                        items_to_return.append(
                            f[sample][component][item][()]
                        )

            return [item/len(samples) for item in items_to_return]

        for sample in samples:
            try:
                item_to_return += f[sample][component][items][()]
            except UnboundLocalError:
                item_to_return = f[sample][component][items][()]

        return item_to_return/len(samples)


def _item_alm_exists(file, component, item):
    """Returns True if component contains alms for the given item, else 
    returns False.
    """
    sample = _get_samples(file)[-1]

    with h5py.File(file, 'r') as f:
        params = list(f[sample][component].keys())

    if f'{item}_alm' in params:
        return True

    return False


def _item_map_exists(file, component, item):
    """Returns True if component contains precomputed map for item, else 
    returns False.
    """
    sample = _get_samples(file)[-1]

    with h5py.File(file, 'r') as f:
        params = list(f[sample][component].keys())

    if f'{item}_map' in params:
        return True

    return False

def _item_exists(file, component, item):
    """Returns True if component contains the item (array or scalar), else 
    returns False.
    """
    sample = _get_samples(file)[-1]

    with h5py.File(file, 'r') as f:
        params = list(f[sample][component].keys())

    if f'{item}' in params:
        return True

    return False


@njit
def unpack_alms_from_chain(data, lmax):
    """Unpacks alms from the Commander chain output.

    Unpacking algorithm: 
    https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1262/src/tools.py#L9

    TODO: look over this function and see if it can be improved.

    Args:
    -----
    data (np.ndarray)
        alms from a commander chainfile.
    lmax : int
        Maximum value for l used in the alms.

    Returns:
    --------
    alms (np.ndarray)
        Unpacked version of the Commander alms (2-dimensional array)
    
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


def chain_to_h5(chainfile, output_dir, nside=None, burn_in=None):
    """Outputs the contents of a chainfile to model hdf5 files for a set of 
    resulutions. 

    Parameters
    ----------
    chainfile : str, `pathlib.PosixPath`
        Path to commander3 hdf5 chainfile.
    output_dir : str, `pathlib.PosixPath`
        Path to where model hdf5 files will be output.
    nside : int
        Custom nside value. Overrides the DEFAULT_NSIDES. Default: None.
    burn_in : Sample at which the chain has "burned in".
    """

    DEFAULT_NSIDES = [
        2**res for res in range(12)     # [1, 2, ... , 2048]
    ]

    NSIDES = DEFAULT_NSIDES if nside is None else nside
    for nside in NSIDES:
        model = model_from_chain(chainfile, nside, burn_in=burn_in)
        model_to_h5(model, output_dir)


def model_to_h5(model, output_dir):
    """Outputs a `cosmoglobe.sky.Model` to a hdf5 model file.

    Parameters
    ----------
    model : `cosmoglobe.sky.Model`
        A cosmoglobe sky model.
    output_dir : str, `pathlib.PosixPath`
        Path to where model hdf5 files will be output.
    """

    dirname = pathlib.Path(output_dir)
    dirname.mkdir(parents=True, exist_ok=True)
    filename = dirname / f'model_{model.nside}.h5'

    with h5py.File(filename, 'w') as f:
        for comp in model:
            grp = f.create_group(comp.label)
            amp = grp.create_dataset('amp', data=comp.amp.value)
            amp.attrs['unit'] = comp.amp.unit.to_string()
            if comp.freq_ref is not None:
                freq = grp.create_dataset('freq_ref', data=comp.freq_ref.value)
                freq.attrs['unit'] = comp.freq_ref.unit.to_string()

            sp_grp = grp.create_group('spectral_parameters')
            for key, value in comp.spectral_parameters.items():
                if isinstance(value, u.Quantity):
                    dset = sp_grp.create_dataset(key, data=value.value)
                    dset.attrs['unit'] = value.unit.to_string()
                else:
                    sp_grp.create_dataset(key, data=value)



def model_from_h5(filename):
    """Initializes a `cosmoglobe.sky.Model` from a hdf5 file with a specific
    format.

    Parameters
    ----------
    filename : str, `pathlib.PosixPath`
        Filename of the hdf5 file from which to read in model.
    """

    filename = pathlib.Path(filename)
    model = Model()
    nside = None
    with h5py.File(filename, 'r') as f:
        for comp in f:
            amp_dset = f.get(f'{comp}/amp')
            amp = u.Quantity(
                value=amp_dset[()], 
                unit=amp_dset.attrs.get('unit', None)
            )
            freq_dset = f.get(f'{comp}/freq_ref')
            if freq_dset:
                freq_ref = u.Quantity(
                    value=freq_dset[()], 
                    unit=freq_dset.attrs.get('unit', None)
                ) 
            else:
                freq_ref = None
            
            spectral_parameters = {}
            for spec in f[comp]['spectral_parameters']:
                dset = f.get(f'{comp}/spectral_parameters/{spec}')
                spectral_parameters[spec] = u.Quantity(
                    value=dset[()],
                    unit=dset.attrs.get('unit', None)
                )

            if nside is None:
                nside = hp.get_nside(amp)

            component = COSMOGLOBE_COMPS[comp]
            if comp == 'radio':
                model._add_component_to_model(
                    component(
                        amp=amp, 
                        freq_ref=freq_ref, 
                        nside=nside,
                        **spectral_parameters
                    )
                )            
            else:
                model._add_component_to_model(
                    component(amp=amp, freq_ref=freq_ref, **spectral_parameters)
                )

    return model