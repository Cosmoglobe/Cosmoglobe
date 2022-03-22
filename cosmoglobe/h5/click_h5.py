import click
from rich import print
import astropy.units as u
import matplotlib.pyplot as plt
import os
import re

from cosmoglobe.h5 import chain

@click.group()
def commands_h5():
    pass

@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.option(
    "-samples",
    default=[-1],
    multiple=True,
    help="",
)
@click.option(
    "-new_name",
    type=click.Path(exists=True),
    default=None,
    help="Name of the chain copy. If None, a default is [chain_name]_copy.h5",
)
@click.option(
    "-min",
    type=click.INT,
    help="Min sample number for range",
)
@click.option(
    "-max",
    type=click.INT,
    help="Max sample number",
)
def copy_chain(**kwargs):
    """Creates a copy of the chain with a single or multiple samples.

    Parameters
    ----------
    chain
        Path to the chain file or the `Chain` object to copy.
    samples
        Samples to copy. Can be an int, a list of ints or a python range object.
    new_name
        Name of the chain copy. If None, a default is "{chain.name}_copy.h5"
    """
    if (kwargs['min'] is not None) or (kwargs['max'] is not None):
        kwargs['samples'] = range(kwargs['min'], kwargs['max'])
    elif kwargs['samples'] == [-1]:
        kwargs['samples']=int(kwargs['samples'][0])
    del kwargs['min']
    del kwargs['max']
    chain.copy_chain(**kwargs)


@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.argument("other_chain", type=click.Path(exists=True),)
@click.option(
    "-group_list",
    multiple=True,
    help="List of hdf5 groups that will be overwritten in the new chainfile. Use -group_list thing1 -group_list thing2.",
)
@click.option(
    "-new_name",
    type=click.Path(exists=True),
    default=None,
    help="Name of the chain copy. If None, a default is [chain name]_copy.h5",
)
def combine_chains(**kwargs):
    """Creates a new chainfile that combines specific groups from two chains.

    The new file will contain all content from `chain`, except for the content
    within the groups in the `group_list`, which are taken from `other_chain`
    instead.

    Parameters
    ----------
    chain
        Path to chain file. This chain defines all the chain whos content you
        want to overwrite in a new combined file.
    other_chain
        Path to chain file.  This chain contains the groups you want to
        overwrite in `chain`.
    group_list
        List of hdf5 groups that will be overwritten in the new chainfile.
    new_name
        Name of the chain copy. If None, a default is "{chain.name}_copy.h5"
    """
    chain.combine_chains(**kwargs)


