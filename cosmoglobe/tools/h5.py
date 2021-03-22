"""
HDH5 routines to handle commander3 chainfiles.
"""
import h5py
import numpy as np


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
                    items_to_return[idx].append(f[sample][component].get(item)[()])

            return [np.mean(item, axis=0) for item in items_to_return]

        for sample in samples:
            try:
                item_to_return = np.append(item_to_return, 
                                           f[sample][component].get(items)[()],
                                           axis=0)
            except UnboundLocalError:
                item_to_return = f[sample][component].get(items)[()]

        return np.mean(item_to_return, axis=0)


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