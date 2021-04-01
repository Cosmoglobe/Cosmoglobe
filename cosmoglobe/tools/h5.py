from .. import sky

from numba import njit
import astropy.units as u
import h5py
import healpy as hp
import numpy as np
import inspect

param_group = 'parameters'  # Model parameter group name as implemented in commander
_ignored_comps = ['md', 'radio', 'relquad'] # These will be dropped from component lists


def model_from_chain(file, sample='mean', burn_in=None):
    """Returns a sky model from a commander3 chainfile.
    
    A cosmoglobe.sky.Model is initialized that represents the sky model used in 
    the given Commander run.

    Args:
    -----
    file (str):
        Path to commander3 hdf5 chainfile.
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
    (cosmoglobe.sky.Model):
        A sky model representing the results of a commander3 run.

    """
    if isinstance(sample, int):
        sample = _int_to_sample(sample)

    component_list = _get_components(file)
    components = []
    for comp in component_list:
        if comp == 'dust' or comp == 'synch': # Delete this once all comps are implemented
            components.append(comp_from_chain(file, comp, sample, burn_in))

    return sky.Model(components, nside=1024)


def comp_from_chain(file, component, sample='mean', burn_in=None):
    """Returns a sky component from a commander3 chainfile.
    
    A sky component that subclasses cosmoglobe.sky.Component is initialized 
    from a given commander run.

    TODO: find better solution to mapping classes to components. Perhaps create 
          a dict that represents the components used in a BP run, e.g:
            BP9_COMPS = {
                'synch': sky.PowerLaw,
                'dust': sky.ModifiedBlackBody,
                ...
            }
          and then pass in BP lvl (8,9,..).
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
        A sky component initialized from a commander run.

    """
    if isinstance(sample, int):
        sample = _int_to_sample(sample)

    comp_classes = {
        'dust': sky.ModifiedBlackBody,
        'synch': sky.PowerLaw,
    }
    comp_class = comp_classes[component]
    args_list = _get_comp_args(comp_class)
    if not 'amp' in args_list or not 'freq_ref' in args_list:
        raise ValueError(
            "component class must contain the arguments 'amp' and 'freq_ref'"
        )
    
    parameters = _get_component_params(file, component)
    freq_ref = (parameters['nu_ref']*u.Hz).to(u.GHz)
    fwhm = (parameters['fwhm']*u.arcmin).to(u.rad)
    nside = parameters['nside']
    args_list.remove('freq_ref')

    unit = parameters['unit']
    if 'k_rj' in unit.lower():
        unit = unit[:-3]
    elif 'k_cmb' in unit.lower():
        unit = unit[:-4]
    unit = u.Unit(unit)

    if sample == 'mean':
        get_items = _get_averaged_items
        sample = _get_samples(file)
        if burn_in is not None:
            sample = sample[burn_in:]
    else:
        get_items = _get_items

    alm_names = []
    map_names = []
    for arg in args_list:
        if _has_precomputed_map(file, component, arg):
            map_names.append(arg)
        else:
            alm_names.append(arg)
    alms = get_items(file, sample, component, [f'{alm}_alm' for alm in alm_names ])
    maps = get_items(file, sample, component, [f'{map_}_map' for map_ in map_names ])

    if isinstance(sample, (tuple, list)):
        sample = sample[-1]

    alm_maps = []
    for idx, alm in enumerate(alms):
        lmax = _get_items(file, sample, component, f'{alm_names[idx]}_lmax')
        unpacked_alm = unpack_alms_from_chain(alm, lmax)
        alms = hp.alm2map(unpacked_alm, 
                          nside=nside, 
                          lmax=lmax, 
                          fwhm=fwhm.value, 
                          verbose=False).astype('float32')
        alm_maps.append(alms)

    args = {}
    args.update({key:value for key, value in zip(alm_names, alm_maps)})
    args.update({key:value for key, value in zip(map_names, maps)})


    args['amp'] *= unit
    args = _set_spectral_units(args)

    return comp_class(comp_name=component, freq_ref=freq_ref, **args)


def _set_spectral_units(maps):
    """    
    TODO: Figure out how to correctly detect unit of spectral map in chain.
          Until then, a hardcoded dict is used:
    """
    units = {
        'T': u.K,
        'Te': u.K,
        'nu_p' : u.GHz,
    }
    for map_ in maps:
        if map_ in units:
            maps[map_] *= units[map_]

    return maps


def _get_comp_args(component_class):
    """Returns a list of arguments needed to initialize a component"""
    ignored_args = ['self', 'comp_name']
    arguments = inspect.signature(component_class.__init__)
    arguments = str(arguments)[1:-1].split(', ')
    return [arg for arg in arguments if arg not in ignored_args]


def _get_samples(file):
    """Returns a list of all samples present in a chain file"""
    with h5py.File(file, 'r') as f:
        samples = list(f.keys())

    samples.remove(param_group)
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
        components = list(f[param_group].keys())

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
        if isinstance(items, (tuple, list)):
            items_to_return = []
            for item in items:
                items_to_return.append(f[sample][component].get(item)[()])

            return items_to_return
        return f[sample][component].get(items)[()]


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
    with h5py.File(file, 'r') as f:
        if isinstance(items, (tuple, list)):
            items_to_return = [[] for _ in range(len(items))]

            for sample in samples:
                for idx, item in enumerate(items):
                    try:
                        items_to_return[idx] += f[sample][component].get(item)[()]
                    except ValueError:
                        items_to_return[idx] = f[sample][component].get(item)[()]

            return [item/len(samples) for item in items_to_return]

        for sample in samples:
            try:
                item_to_return += f[sample][component].get(items)[()]
            except UnboundLocalError:
                item_to_return = f[sample][component].get(items)[()]

        return item_to_return/len(samples)


def _has_precomputed_map(file, component, item, sample=-1):
    """Returns True if component contains precomputed map for item, else 
    returns False.
    """
    if sample == -1:
        sample = _get_samples(file)[-1]

    with h5py.File(file, 'r') as f:
        params = list(f[sample][component].keys())

    if f'{item}_map' in params:
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
