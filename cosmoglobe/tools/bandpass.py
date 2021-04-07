from cosmoglobe.tools.map import StokesMap
import numpy as np


def _extract_scalars(iterator):
    """Returns all scalars from an iterator.
    
    A scalar here refers to either an int, float or array filled with a 
    single value.

    Args:
    -----
    iterator : tuple, list, dict
        Iterator object. Can be any object with the method __iter__ defined.

    Returns:
    --------
    scalars : tuple, list, dict
        Iterator containing extracted scalars. The returned iterator will 
        match the type of the input iterator. If no scalars are present, 
        returns None

    """
    if isinstance(iterator, (tuple, list)):
        scalars = []
        for value in iterator:
            uniques = np.unique(value.data)
            if len(uniques) == 1:
                scalar = uniques[0]
                scalars.append(scalar)
        
        if scalars:
            if isinstance(iterator, tuple):
                return tuple(scalars)

            return scalars
    
    if isinstance(iterator, dict):
        scalars = {}
        for key, value in iterator.items():
            uniques = np.unique(value.data)
            if len(uniques) == 1:
                scalar = uniques[0]
                scalars[key] = scalar

        if scalars:
            return scalars

    return


