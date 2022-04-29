import click
from rich import print
import healpy as hp
import numpy as np

from cosmoglobe.h5 import chain
from cosmoglobe.h5.chain import Chain

@click.group()
def commands_h5():
    pass

# fmt: off
@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.option( "-samples", default=[-1], multiple=True, help="",)
@click.option( "-new_name", type=click.Path(exists=True), default=None, help="Name of the chain copy. If None, a default is [chain_name]_copy.h5",)
@click.option( "-min", type=click.INT, help="Min sample number for range",)
@click.option( "-max", type=click.INT, help="Max sample number",)
# fmt: on
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

# fmt: off
@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.argument("other_chain", type=click.Path(exists=True),)
@click.option( "-group_list", multiple=True, help="List of hdf5 groups that will be overwritten in the new chainfile. Use -group_list thing1 -group_list thing2.",)
@click.option( "-new_name", type=click.Path(exists=True), default=None, help="Name of the chain copy. If None, a default is [chain name]_copy.h5",)
# fmt: on
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

# fmt: off
@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.argument("dataset",)
@click.argument("outname", )
@click.option( "-min", type=click.INT, default=0, help="First sample.",)
@click.option( "-max", type=click.INT, default=None,  help="Last sample. If not specified, use last sample of dataset.",)
@click.option( "-nside", type=click.INT, default=None,  help="Output nside of data.",)
# fmt: on
def mean(chain, dataset, outname, min, max, nside):
    """Calculates the mean of a dataset over a range of samples

    Parameters
    ----------
    chain
        Path to chain file. This chain defines all the chain whos content you
        want to overwrite in a new combined file.
    dataset
        Chain dataset on the form dust/amp_alm
    outname
        Name of .fits file to output the result to.
    min
        The first sample, default is first sample of chain.
    max
        last sample, default is last sample of chain.
    nside
        Output nside of data.
    """

    chain = Chain(chain)
    if max is None: max = chain.nsamples
    data = chain.mean(dataset, samples=range(min, max))

    if outname.endswith(".fits"):
        if dataset.endswith("alm"): 
            comp, quantity = dataset.split("/")
            if nside is None: nside = chain.parameters[comp]["nside"]
            pol = True if quantity.startswith("amp") else False
            fwhm = chain.parameters[comp]["fwhm"]
            data = hp.alm2map(data, nside=nside, fwhm=fwhm, pol=pol, pixwin=True,)
            
        hp.write_map(outname, data, overwrite=True)
    else:
        np.savetxt(outname, data)

# fmt: off
@commands_h5.command()
@click.argument("chain", type=click.Path(exists=True),)
@click.argument("dataset",)
@click.argument("outname", )
@click.option( "-min", type=click.INT, default=0, help="First sample.",)
@click.option( "-max", type=click.INT, default=None,  help="Last sample. If not specified, use last sample of dataset.",)
@click.option( "-nside", type=click.INT, default=None,  help="Output nside of data.",)
# fmt: on
def stddev(chain, dataset, outname, min, max, nside):
    """Calculates the stddev of a dataset over a range of samples

    Parameters
    ----------
    chain
        Path to chain file. This chain defines all the chain whos content you
        want to overwrite in a new combined file.
    dataset
        Chain dataset on the form dust/amp_alm
    outname
        Name of .fits file to output the result to.
    min
        The first sample, default is first sample of chain.
    max
        last sample, default is last sample of chain.
    nside
        Output nside of data.
    """

    chain = Chain(chain)
    if max is None: max = chain.nsamples

    if outname.endswith(".fits"):
        alm2map = True if dataset.endswith("alm") else False
        data = chain.stddev(dataset, samples=range(min, max), alm2map=alm2map)
        hp.write_map(outname, data, overwrite=True)
    else:
        data = chain.stddev(dataset, samples=range(min, max),)
        np.savetxt(outname, data)

