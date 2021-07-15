.. cosmoglobe.sky:

Base classes for sky components
===============================

.. currentmodule:: cosmoglobe.sky.base

All sky components inherit from either ``DiffuseComponent``,
``PointSourceComponent``, or ``LineComponent``, which all inherit from
``Component``. These classes provide common methods and attributes shared among
the three component types which dictates how the sky emission of a given
component is computed for a single frequency or when integrated over a bandpass.

.. autosummary:: 
    :toctree: generated/

    Component
    DiffuseComponent
    PointSourceComponent
    LineComponent
    
