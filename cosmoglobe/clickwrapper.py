import click

"""
THIS IS ALL CURRENTLY PSEUDOCODE
"""

@commands_plotting.command()
@click.argument("filename", type=click.STRING)
def plotfits(filename):
    """
    This function wraps around the plotting function and plots fits files
    input: filename of fits file, optional plotting params
    """
    from cosmoglobe.plot.mollweide import mollplot
    import fitsio
    data = fitsio.open(filename)
    map_ = filename.to_map()
    mollplot(map_)

@commands_plotting.command()
@click.argument("filename", type=click.STRING)
def plothdf(filename, dataset):
    """
    This function wraps around the plotting function and plots hdf files
    Input: hdf file, dataset and nside, optional plotting params
    """
    from cosmoglobe.plot.mollweide import mollplot
    import h5
    data = h5.open(filename, dataset)
    map_ = filename.to_map()
    mollplot(map_)
