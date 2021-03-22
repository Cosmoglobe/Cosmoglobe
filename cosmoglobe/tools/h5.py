"""
HDH5 routines to handle commander3 chainfiles.
"""
import h5py
import numpy as np



param_group = 'parameters'
test_file = '/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/bla.h5'



def _get_samples(file):
    """Returns a list of all samples present in a chain file"""
    with h5py.File(file, 'r') as f:
        samples = list(f.keys())

    samples.remove(param_group)
    return samples


def _get_components(file):
    """Returns a list of all components present in a chain file"""
    with h5py.File(file, 'r') as f:
        components = list(f[param_group].keys())

    return components


def _get_items(file, sample, component, items):
    """Return the value of one or many items in the chain file.

    Args:
    -----
    file: str
        Path to h5 file.
    sample : str
        sample name.
    component: str
        Component group name.
    items: str, list
        Name of item to extract, or a list of names.

    Returns:
    --------
    lists
        list of items extracted from the chain file.
    
    """
    with h5py.File(file, 'r') as f:
        if isinstance(items, (tuple, list)):
            items_to_return = []
            for item in items:
                items_to_return.append(f[sample][component].get(item)[()])

            return items_to_return
        
        return f[sample][component].get(items)[()]


def _get_averaged_items(file, samples, component, items):
    """Return the averaged value of one or many item in the chain file.

    Args:
    -----
    file: str
        Path to h5 file.
    samples : list
        List of samples to average over.
    component:
        Component group.
    items: str, list
        Name of item to extract, or a list of names. Items must be of types
        compatible with integer division.
    samples: list
        List of all samples to average over.

    Returns:
    --------
    lists
        list of items extracted from the chain file.
    
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



if __name__ == '__main__':
    samples = _get_samples(test_file)
    items = _get_averaged_items(test_file,
                                samples,
                                'ame', 
                                ('amp_alm', 'nu_p_map'))
    print(items)
    items = _get_items(test_file, 
                       '000039',
                       'ame', 
                       ('amp_alm', 'nu_p_map'))
    print(items)