import time
import os
import numpy as np
import sys
import click
from cosmoglobe.release.tools import *

@click.group()
def commands_fits():
    pass

@commands_fits.command()
@click.argument("input", type=click.STRING)
def printheader(input,):
    """
    Prints the header of a fits file.
    """
    from astropy.io import fits

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
    from astropy.io import fits

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
    from astropy.io import fits

    with fits.open(input) as hdulist:
        for hdu in hdulist:
            hdu.header.pop(columns)        
            hdu.data.del_col(columns)        
        hdulist.writeto(output, overwrite=True)
        
@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.option("-mask", type=click.STRING, help="Mask for dipole removal, 30deg sky cut by default")
@click.option("-sig", type=click.INT, multiple=True, help="fields to calculate, [0,1,2 by default]")
def rmmd(input, mask, sig):
    """
    removes the dipole and mask of input file
    if mask = "auto" uses 30 deg sky cut
    """
    import healpy as hp
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
            mono, dip = hp.fit_dipole(m_masked[i], gal_cut=30)            
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


@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
def QU2ang(input, output,):
    """
    Calculates polarization angle map from QU signals.
    """
    import healpy as hp
    Q, U = hp.read_map(input, field=(1,2), dtype=None, verbose=False)
    phi = 0.5*np.arctan(U,Q)
    hp.write_map(output, phi, dtype=None, overwrite=True)


@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("template", nargs=-1, type=click.STRING)
@click.option("-mask", type=click.STRING)
@click.option("-noise", type=click.STRING)
@click.option("-res", type=click.STRING)
def fittemp(input, template, mask, noise, res):
    """
    Calculates polarization angle map from QU signals.
    """
    import healpy as hp
    field = (1,2)
    map_ = hp.read_map(input, field=(0,1), dtype=None, verbose=False)
    nside = hp.get_nside(map_)
    npix = hp.nside2npix(nside)
    ntemps = len(template)
    temp = np.zeros((ntemps, *map_.shape))
    #print("input: ", input, "template: ", template, "mask: ", mask)
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
        from astropy.io import fits
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




@commands_fits.command()
@click.argument("input1", type=click.STRING)
@click.argument("input2", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-beam1", type=click.STRING, help="Optional beam file for input 1",)
@click.option("-beam2", type=click.STRING, help="Optional beam file for input 2",)
@click.option("-mask", type=click.STRING, help="Mask",)
def crosspec(input1, input2, output, beam1, beam2, mask,):
    """
    Calculates a powerspectrum from polspice.
    Using this path /mn/stornext/u3/trygvels/PolSpice_v03-03-02/
    """
    sys.path.append("/mn/stornext/u3/trygvels/PolSpice_v03-03-02/")
    from ispice import ispice

    lmax = 6000
    fwhm = 0
    #mask = "dx12_v3_common_mask_pol_005a_2048_v2.fits"
    if beam1 and beam2:
        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam_file1=beam1,
               beam_file2=beam2,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )
    else:

        ispice(input1,
               clout=output,
               nlmax=lmax,
               beam1=0.0,
               beam2=0.0,
               mapfile2=input2,
               maskfile1=mask,
               maskfile2=mask,
               fits_out="NO",
               polarization="YES",
               subav="YES",
               subdipole="YES",
               symmetric_cl="YES",
           )

@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-minchain", default=1, help="lowest chain number, c0002 [ex. 2] (default=1)",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5] (default=1)",)
@click.option("-chaindir", default=None, type=click.STRING, help="Base of chain directory, overwrites chain iteration from input file name to iteration over chain directories, BP_chain_c15 to BP_chain_c19 [ex. 'BP_chain', with minchain = 15 and maxchain = 19]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_mean(
        input, output, min, max, minchain, maxchain, chaindir, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the mean over sample range from fits-files.
    ex. res_030_c0001_k000001.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3\n
    If output name is set to .dat, data will not be converted to map.

    Note: the input file name must have the 'c0001' chain identifier and the 'k000001' sample identifier. The -min/-max and -chainmin/-chainmax options set the actual samples/chains to be used in the calculation 
    """

    fits_handler(input, min, max, minchain, maxchain, chaindir, output, fwhm, nside, zerospin, missing, pixweight, np.mean, write=True)

@commands_fits.command()
@click.argument("input", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-minchain", default=1, help="lowest chain number, c0002 [ex. 2] (default=1)",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5] (default=1)",)
@click.option("-chaindir", default=None,type=click.STRING, help="Base of chain directory, overwrites chain iteration from input file name to iteration over chain directories, BP_chain_c15 to BP_chain_c19 [ex. 'BP_chain', with minchain = 15 and maxchain = 19]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for down-grading maps before calculation",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-missing", is_flag=True, help="If files are missing, drop them. Else, exit computation",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def fits_stddev(
        input, output, min, max, minchain, maxchain, chaindir, fwhm, nside, zerospin, missing, pixweight):
    """
    Calculates the standard deviation over sample range from fits-files.
    ex. res_030_c0001_k000001.fits res_030_20-100_mean_40arcmin.fits -min 20 -max 100 -fwhm 40 -maxchain 3
    If output name is set to .dat, data will not be converted to map.

    Note: the input file name must have the 'c0001' chain identifier and the 'k000001' sample identifier. The -min/-max and -chainmin/-chainmax options set the actual samples/chains to be used in the calculation 
    """

    fits_handler(input, min, max, minchain, maxchain, chaindir, output, fwhm, nside, zerospin, missing, pixweight, np.std, write=True)
