import click
from rich import print
from astropy.io import fits

import healpy as hp
import numpy as np


@click.group()
def commands_fits():
    pass

@commands_fits.command()
@click.argument("input", type=click.STRING)
def printheader(input,):
    """
    Prints the header of a fits file.
    """
    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.header))

@commands_fits.command()
@click.argument("input", type=click.STRING)
def printdata(input,):
    """
    Prints the data of a fits file
    """
    with fits.open(input) as hdulist:
        hdulist.info()
        for hdu in hdulist:
            print(repr(hdu.data))


@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.argument("columns", type=click.INT)
def rmcolumn(input, output, columns):
    """
    Removes columns in fits file
    """
    with fits.open(input) as hdulist:
        for hdu in hdulist:
            hdu.header.pop(columns)        
            hdu.data.del_col(columns)        
        hdulist.writeto(output, overwrite=True)
        
# fmt: off
@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.option("-mask", type=click.STRING, help="Mask for dipole removal, 30deg sky cut by default")
@click.option("-gal_cut", default=30, type=click.FLOAT, help="Latitude cut if no mask is provided")
@click.option("-sig", type=click.INT, multiple=True, help="fields to calculate, [0,1,2 by default]")
# fmt: on
def rmmd(input, mask, sig, gal_cut):
    """
    removes the dipole and mask of input fits file
    If mask is not specified, uses 30
    """

    try:
        m = hp.read_map(input, field=(0,1,2), verbose=False, dtype=None,)
    except:
        m = hp.read_map(input, field=(0,), verbose=False, dtype=None,)
    nside = hp.get_nside(m)
    npix = hp.nside2npix(nside)
    print(f"Nside: {nside}, npix: {npix}")

    # Mask map for dipole estimation
    m_masked = hp.ma(m)
    if mask:
        m_masked.mask = np.logical_not(hp.read_map(mask,verbose=False,dtype=None,))
    
    # Fit dipole to masked map
    for i in sig:
        if not mask:
            mono, dip = hp.fit_dipole(m_masked[i], gal_cut=gal_cut)            
        else:
            mono, dip = hp.fit_dipole(m_masked[i])
    
        # Subtract dipole map from data
        click.echo(click.style("Removing dipole:", fg="yellow"))
        click.echo(click.style("Dipole vector:",fg="green") + f" {dip}")
        click.echo(click.style("Dipole amplitude:",fg="green") + f" {np.sqrt(np.sum(dip ** 2))}")
        click.echo(click.style("Monopole:",fg="green") + f" {mono}")
        # Create dipole template
        ray = range(npix)
        vecs = hp.pix2vec(nside, ray)
        dipole = np.dot(dip, vecs)
        m[i] = m[i] - dipole - mono
    hp.write_map(input.replace(".fits", "_no-md.fits"), m, dtype=None, overwrite=True)


# fmt: off
@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("template", nargs=-1, type=click.STRING)
@click.option("-mask", type=click.STRING)
@click.option("-noise", type=click.STRING)
@click.option("-res", type=click.STRING)
# fmt: on
def fittemp(input, template, mask, noise, res):
    """
    Takes a fits map and fits the templates to it. 
    Outputs template fit values and uncertainty
    """
    field = (1,2)
    map_ = hp.read_map(input, field=(0,1), dtype=None, verbose=False)
    nside = hp.get_nside(map_)
    npix = hp.nside2npix(nside)
    ntemps = len(template)
    temp = np.zeros((ntemps, *map_.shape))

    for i in range(ntemps):
        temp[i] = hp.read_map(template[i], field=field, dtype=None, verbose=False)
    if mask:
        mask = np.logical_not(hp.read_map(mask, field=field, dtype=None, verbose=False))
    else:
        mask = np.logical_not(np.ones_like(map_))

    map_masked = hp.ma(map_)
    map_masked.mask = mask
    temp_masked = hp.ma(temp)
    temp_masked.mask = mask

    if noise: #noise
        with fits.open(noise) as hdulist:
            for hdu in hdulist:
                N = hdu.data
                N *= 1e-6
                print("scaling cov by 1e3")
    else:
        N = np.eye(2*npix)
    print("map:", map_.shape, "template:", temp.shape, "mask:", mask.shape, "N:", N.shape)
    t = temp_masked.reshape((ntemps, 2*npix)).T
    a = np.linalg.inv((t.T).dot(N).dot(t))
    print(a.shape)
    err = np.sqrt(np.diagonal(a))
    a = a.dot(t.T).dot(N).dot(map_masked.ravel())
    for i in range(len(a)):
        print(f"T_{i}: {a[i]:.3f} +- {err[i]:.5f}")
    residual = map_masked - np.sum(temp_masked*a.reshape(ntemps,1,1), axis=0)
    residual_masked = hp.ma(residual)
    residual_masked.mask = mask
    print("Sum of residual: ", np.sum(residual_masked))

    if res:
        hp.write_map(res, residual_masked, dtype=np.float32, overwrite=True)
