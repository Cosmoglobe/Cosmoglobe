"""
HDH5 routines to handle commander3 chainfiles.
"""
from .. import sky

import astropy.units as u
import h5py
import numpy as np
import inspect

param_group = 'parameters'  # Model parameter group name as implemented in commander
_ignored_comps = ['md', 'radio', 'relquad'] # These will be dropped from component lists

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


def _get_comp_args(component_class):
    ignored_args = ['self', 'comp_name']
    arguments = inspect.signature(component_class.__init__)
    arguments = str(arguments)[1:-1].split(', ')
    return [arg for arg in arguments if arg not in ignored_args]



def comp_from_chain(file, component, sample='mean', burn_in=None):
    """Returns an initialized sky component from a chainfile.

    TODO: Figure out how to correctly detect unit of spectral map in chain

    """
    args_list = _get_comp_args(sky.ModifiedBlackBody)
    if not 'amp' in args_list or not 'freq_ref' in args_list:
        raise ValueError(
            "component class must contain the arguments 'amp' and 'freq_ref'"
        )
    
    parameters = _get_component_params(file, component)

    freq_ref = parameters['nu_ref']*u.Hz
    freq_ref = freq_ref.to(u.GHz)
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

    args = {}

    alms = get_items(file, sample, component, [f'{alm}_alm' for alm in alm_names ])
    maps = get_items(file, sample, component, [f'{map_}_map' for map_ in map_names ])

    args.update({key:value for key, value in zip(alm_names, alms)})
    args.update({key:value for key, value in zip(map_names, maps)})

    args['amp'] *= unit

    print(args)

def model_from_chain(file):
    """Returns an initialized sky component from a chainfile."""
    component_list = _get_components(file)
    components = {}
    for comp in component_list:
        params = _get_component_params(file, comp)
        print(params)
        if comp == 'synch':
            components['synch'] = sky.PowerLaw
        elif comp == 'dust':
            components['dust'] = sky.ModifiedBlackBody

    # for comp_name, comp_class in components.items():

        # amp, freq_ref, spectrals = _get_averaged_items(file, samples, comp_name, spectral_args)
        


if __name__ == '__main__':
    test_file = '/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/bla.h5'
    samples = _get_samples(test_file)
    # items = _get_averaged_items(test_file,
    #                             samples,
    #                             'ame', 
    #                             ('amp_alm', 'nu_p_map'))
    # print(items)
    # items = _get_items(test_file, 
    #                    '000039',
    #                    'ame', 
    #                    ('amp_alm', 'nu_p_map'))
    # print(items)
    # print(_get_components(test_file, ignore_comps=False))
    print(_has_precomputed_map(test_file, 'ff', 'Te'))