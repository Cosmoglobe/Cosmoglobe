.. cosmoglobe.sky:

Sky components
==============

.. currentmodule:: cosmoglobe

Classes representing the components in the Cosmoglobe Sky Model. A component is
essentially defined by its ``get_freq_ref`` method, which returns the scaling
factor :math:`f_\mathrm{comp}(\nu)`. The ``__call__`` method of a component will
return the sky emission of that component, similarly to how ``cosmoglobe.Model()``
returns the sum of all component emission in the model.

.. autosummary:: 
    :toctree: generated/
    
    AME
    CMB
    Dust
    FreeFree
    Radio
    Synchrotron