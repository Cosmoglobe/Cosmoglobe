.. cosmoglobe.sky.components:

Sky components
==============

.. currentmodule:: cosmoglobe.sky.components

Classes representing the components in the Cosmoglobe Sky Model. The
``__call__`` method of a component will return the sky emission of that
component, similarly to how ``Model()`` returns the sum of all component
emission in the model.

.. autosummary:: 
    :toctree: generated/
    
    ame.SpinningDust
    cmb.CMB
    dust.ModifiedBlackbody
    freefree.LinearOpticallyThin
    radio.AGNPowerLaw
    synchrotron.PowerLaw