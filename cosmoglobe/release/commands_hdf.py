import time
import os
import numpy as np
import sys
import click
from cosmoglobe.release.tools import *
from cosmoglobe.release.commands_plotting import *
from cosmoglobe.release.fitsformatter import format_fits, get_data, get_header

@click.group()
def commands_hdf():
    pass

@commands_hdf.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-notchain", is_flag=True, help="Flag if parsing a non-chain hdf file",)
def split(input,dataset,output,min,max,maxchain,notchain):
    """
    This function saves whatever specified data to a separate file.
    """
    h5handler(input, dataset, min, max, maxchain, output, fwhm=None, nside=None, command=False, pixweight=None, zerospin=False, lowmem=False,notchain=True)

@commands_hdf.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for alm binning",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def mean(input, dataset, output, min, max, maxchain, fwhm, nside, zerospin, pixweight):
    """
    Calculates the mean over sample range from .h5 file.
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -maxchain 3
    ex. chains_c0001.h5 dust/amp_alm 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -nside 512
    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()


    h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.mean, pixweight, zerospin,)

@commands_hdf.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("output", type=click.STRING)
@click.option("-min", default=1, type=click.INT, help="Start sample, default 1",)
@click.option("-max", default=None, type=click.INT, help="End sample, calculated automatically if not set",)
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-fwhm", default=0.0, help="FWHM in arcmin")
@click.option("-nside", default=None, type=click.INT, help="Nside for alm binning",)
@click.option("-zerospin", is_flag=True, help="If smoothing, treat maps as zero-spin maps.",)
@click.option("-pixweight", default=None, type=click.STRING, help="Path to healpy pixel weights.",)
def stddev(input, dataset, output, min, max, maxchain, fwhm, nside, zerospin, pixweight,):
    """
    Calculates the stddev over sample range from .h5 file.
    ex. chains_c0001.h5 dust/amp_map 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -maxchain 3
    ex. chains_c0001.h5 dust/amp_alm 5 50 dust_5-50_mean_40arcmin.fits -fwhm 40 -nside 512

    If output name is set to .dat, data will not be converted to map.
    """
    if dataset.endswith("alm") and nside == None:
        click.echo("Please specify nside when handling alms.")
        sys.exit()

    h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.std, pixweight, zerospin,)

@commands_hdf.command()
@click.argument("filename", type=click.STRING)
@click.argument("nchains", type=click.INT)
@click.argument("burnin", type=click.INT)
@click.option("-suffix", type=click.STRING, default="",  help='if there is some, ex. Tresamp_v1')
@click.option("-path", default="cmb/sigma_l", help="Dataset path ex. cmb/sigma_l",)
@click.argument("outname", type=click.STRING)
def sigma_l2fits(filename, nchains, burnin, path, suffix, outname, save=True,):
    """
    Converts c3-h5 dataset to fits for c1 BR and GBR estimator analysis.
    ex. c3pp sigma-l2fits chains_v1/chain 5 10 cmb_sigma_l_GBRlike.fits
    If "chain_c0001.h5", filename is cut to "chain" and will look in same directory for "chain_c*****.h5". If a -suffix is present, it is added after c****, ex. -suffix Tr\
esamp_v1 will point to chain_c****_Tresamp_v1.h5.
    See comm_like_tools for further information about BR and GBR post processing
    """
    click.echo("{:-^48}".format("Formatting sigma_l data to fits file"))
    import h5py
    if filename.endswith(".h5"):
        filename = filename.rsplit("_", 1)[0]
    temp = np.zeros(nchains)
    for nc in range(1, nchains + 1):
        with h5py.File(filename + "_c" + str(nc).zfill(4) + suffix + ".h5", "r",) as f:
            groups = list(f.keys())
            temp[nc - 1] = len(groups)
    nsamples_max = int(max(temp[:]))
    click.echo(f"Largest chain has {nsamples_max} samples, using burnin {burnin}\n")
    for nc in range(1, nchains + 1):
        fn = filename + "_c" + str(nc).zfill(4) + suffix + ".h5"
        with h5py.File(fn, "r",) as f:
            click.echo(f"Reading {fn}")
            groups = list(f.keys())
            nsamples = len(groups)
            if nc == 1:
                dset = np.zeros((nsamples_max + 1, 1, len(f[groups[0] + "/" + path]), len(f[groups[0] + "/" + path][0]),))
                nspec = len(f[groups[0] + "/" + path])
                lmax = len(f[groups[0] + "/" + path][0]) - 1
            else:
                dset = np.append(dset, np.zeros((nsamples_max + 1, 1, nspec, lmax + 1,)), axis=1,)
            click.echo(f"Dataset: {path} \n# samples: {nsamples} \n# spectra: {nspec} \nlmax: {lmax}")
            for i in range(nsamples):
                for j in range(nspec):
                    dset[i + 1, nc - 1, j, :] = np.asarray(f[groups[i] + "/" + path][j][:])
            click.echo("")
    # Optimize with jit?
    ell = np.arange(lmax + 1)
    for nc in range(1, nchains + 1):
        for i in range(1, nsamples_max + 1):
            for j in range(nspec):
                dset[i, nc - 1, j, :] = dset[i, nc - 1, j, :] * ell[:] * (ell[:] + 1.0) / 2.0 / np.pi
    dset[0, :, :, :] = nsamples - 2 #burnin 

    if save:
        click.echo(f"Dumping fits file: {outname}")
        dset = np.asarray(dset, dtype="f4")

        from astropy.io import fits
        head = fits.Header()
        head["FUNCNAME"] = ("Gibbs sampled power spectra",  "Full function name")
        head["LMAX"]     = (lmax,  "Maximum multipole moment")
        head["NUMSAMP"]  = (nsamples_max,  "Number of samples")
        head["NUMCHAIN"] = (nchains,  "Number of independent chains")
        head["NUMSPEC"]  = (nspec,  "Number of power spectra")
        fits.writeto(outname, dset, head, overwrite=True)                

        # FITSIO Saving Deprecated (Use astropy)
        if False:
            import fitsio

            fits = fitsio.FITS(outname, mode="rw", clobber=True, verbose=True,)
            h_dict = [
                {"name": "FUNCNAME", "value": "Gibbs sampled power spectra", "comment": "Full function name",},
                {"name": "LMAX", "value": lmax, "comment": "Maximum multipole moment",},
                {"name": "NUMSAMP", "value": nsamples_max, "comment": "Number of samples",},
                {"name": "NUMCHAIN", "value": nchains, "comment": "Number of independent chains",},
                {"name": "NUMSPEC", "value": nspec, "comment": "Number of power spectra",},
            ]
            fits.write(dset[:, :, :, :], header=h_dict, clobber=True,)
            fits.close()

    return dset

def h5map2fits(filename, dataset, save=True):
    """
    Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
    """
    import healpy as hp
    import h5py

    dataset, tag = dataset.rsplit("/", 1)

    with h5py.File(filename, "r") as f:
        maps = f[f"{dataset}/{tag}"][()]
        lmax = f[f"{dataset}/amp_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    dataset = f"{dataset}/{tag}"
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    if save:
        hp.write_map(outfile + f"_n{str(nside)}.fits", maps, overwrite=True,dtype=None)
    return maps, nside, lmax, outfile

@commands_hdf.command()
@click.argument("filename", type=click.STRING)
@click.argument("dataset", type=click.STRING)
def h52fits(filename, dataset,):
    """
    Outputs a .h5 map to fits on the form 000001_cmb_amp_n1024.fits
    """
    import healpy as hp
    import h5py

    dataset, tag = dataset.rsplit("/", 1)

    with h5py.File(filename, "r") as f:
        maps = f[f"{dataset}/{tag}"][()]
        if ('aml' in tag):
            lmax = f[f"{dataset}/amp_lmax"][()]  # Get lmax from h5

    nside = hp.npix2nside(maps.shape[-1])
    dataset = f"{dataset}/{tag}"
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_map", "")
    hp.write_map(outfile + f"_n{str(nside)}.fits", maps, overwrite=True,dtype=None)
    

@commands_hdf.command()
@click.argument("input", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.argument("nside", type=click.INT)
@click.option("-lmax", default=None, type=click.INT)
@click.option("-fwhm", default=0.0, type=click.FLOAT)
def alm2fits(input, dataset, nside, lmax, fwhm):
    """
    Converts c3 alms in .h5 file to fits.
    Specify nside and optional smoothing.
    """
    alm2fits_tool(input, dataset, nside, lmax, fwhm)


@commands_hdf.command()
@click.argument("chain", type=click.Path(exists=True), nargs=-1,)
@click.argument("burnin", type=click.INT)
@click.argument("procver", type=click.STRING)
@click.option("-resamp", is_flag=True, help="data interpreted as resampled data",)
@click.option("-copy", "copy_", is_flag=True, help=" copy full .h5 file",)
@click.option("-freqmaps", is_flag=True, help=" output freqmaps",)
@click.option("-ame", is_flag=True, help=" output ame",)
@click.option("-ff", "-freefree", "ff", is_flag=True, help=" output freefree",)
@click.option("-cmb", is_flag=True, help=" output cmb",)
@click.option("-synch", is_flag=True, help=" output synchrotron",)
@click.option("-dust", is_flag=True, help=" output dust",)
@click.option("-br", is_flag=True, help=" output BR",)
@click.option("-diff", is_flag=True, help="Creates diff maps official releases")
@click.option("-diffcmb", is_flag=True, help="Creates diff maps cmb")
@click.option("-goodness", is_flag=True, help="Output chisq and residual maps in separate dir")
@click.option("-chisq", is_flag=True, help="Output chisq ")
@click.option("-res", is_flag=True, help="Output residuals")
@click.option("-all", "all_", is_flag=True, help="Output all")
@click.option("-plot", is_flag=True, help="Plot everything (invoke plotrelease)")
@click.option("-pol", is_flag=True, help="if resamp is pol or T")
@click.pass_context
def release(ctx, chain, burnin, procver, resamp, copy_, freqmaps, ame, ff, cmb, synch, dust, br, diff, diffcmb, goodness, chisq, res, all_, plot, pol):
    """
    Creates a release file-set on the BeyondPlanck format.
    https://gitlab.com/BeyondPlanck/repo/-/wikis/BeyondPlanck-Release-Candidate-2

    ex. c3pp release chains_v1_c{1,2}/chain_c000{1,2}.h5 30 BP_r1
    Will output formatted files using all chains specified,
    with a burnin of 30 to a directory called BP_r1

    This function outputs the following files to the {procver} directory:
    BP_chain01_{procver}.h5
    BP_resamp_chain01_Cl_{procver}.h5
    BP_resamp_chain01_noCl_{procver}.h5
    BP_param_v1.txt
    BP_param_resamp_Cl_v1.txt
    BP_param_resamp_noCl_v1.txt

    BP_030_IQU_n0512_{procver}.fits
    BP_044_IQU_n0512_{procver}.fits
    BP_070_IQU_n1024_{procver}.fits

    BP_cmb_IQU_n1024_{procver}.fits
    BP_synch_IQU_n1024_{procver}.fits
    BP_freefree_I_n1024_{procver}.fits
    BP_ame_I_n1024_{procver}.fits

    BP_cmb_GBRlike_{procver}.fits
    """
    # TODO
    # Use proper masks for output of CMB component
    # Use inpainted data as well in CMB component

    from pathlib import Path
    import shutil

    if all_: # sets all other flags to true
        copy_ = not copy_; freqmaps = not freqmaps; ame = not ame; ff = not ff; cmb = not cmb
        synch = not synch; dust = not dust; br = not br; diff = not diff; diffcmb = not diffcmb
        goodness = not goodness; res = not res; chisq = not chisq; plot = not plot

    if goodness:
        chisq = res = True
    elif chisq or res:
        goodness = True

    # Make procver directory if not exists
    click.echo("{:#^80}".format(""))
    click.echo(f"Creating directory {procver}")
    Path(procver).mkdir(parents=True, exist_ok=True)
    chains = chain
    maxchain = len(chains)

    if pol: # Dumb labeling thing
        pol="P"
    else:
        pol="T"
    """
    Copying chains files
    """
    if copy_:
        # Commander3 parameter file for main chain
        for i, chainfile in enumerate(chains, 1):
            path = os.path.split(chainfile)[0]
            for file in os.listdir(path):
                if file.startswith("param") and i == 1:  # Copy only first
                    click.echo(f"Copying {path}/{file} to {procver}/BP_param_c" + str(i).zfill(4) + f"_{procver}.txt")
                    if resamp:
                        shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_c" + str(i).zfill(4) + f"_{pol}resamp_{procver}.txt")
                    else:
                        shutil.copyfile(f"{path}/{file}", f"{procver}/BP_param_c" + str(i).zfill(4) + f"_{procver}.txt")

            if resamp:
                # Resampled CMB-only full-mission Gibbs chain file with Cls (for BR estimator)
                click.echo(f"Copying {chainfile} to {procver}/BP_c" + str(i).zfill(4) + f"_{pol}resamp_{procver}.h5")
                shutil.copyfile(chainfile, f"{procver}/BP_c" + str(i).zfill(4) + f"_{pol}resamp_{procver}.h5",)
            else:
                # Full-mission Gibbs chain file
                click.echo(f"Copying {chainfile} to {procver}/BP_c" + str(i).zfill(4) + f"_{procver}.h5")
                shutil.copyfile(chainfile, f"{procver}/BP_c" + str(i).zfill(4) + f"_{procver}.h5",)

     #if halfring:
     #   # Copy halfring files
     #   for i, chainfile in enumerate([halfring], 1):
     #       # Copy halfring files
     #       click.echo(f"Copying {resamp} to {procver}/BP_halfring_c" + str(i).zfill(4) + f"_{procver}.h5")
     #       shutil.copyfile(halfring, f"{procver}/BP_halfring_c" + str(i).zfill(4) + f"_{procver}.h5",)

    """
    IQU mean, IQU stdev, (Masks for cmb)
    Run mean and stddev from min to max sample (Choose min manually or start at 1?)
    """
    if resamp:
        chain = f"{procver}/BP_c0001_{pol}resamp_{procver}.h5"
    else:
        chain = f"{procver}/BP_c0001_{procver}.h5"
    if freqmaps:
        try:
            # Ideally, we would have a way to access this information from the
            # chain itself, but currently we aren't outputting any of the main
            # stuff from the parameter file.
            # A lot of the nu_ref, bandctr, restfreq, bndwid parameters are just
            # dummies for now.
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="023-WMAP_K",
                fwhm=0.0,
                nu_ref_t="23 GHz",
                nu_ref_p="23 GHz",
                procver=procver,
                filename=f"BP_023-WMAP_K_IQU_n0512_{procver}.fits",
                bndctr=23,
                restfreq=23,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="030-WMAP_Ka",
                fwhm=0.0,
                nu_ref_t="30 GHz",
                nu_ref_p="30 GHz",
                procver=procver,
                filename=f"BP_030-WMAP_Ka_IQU_n0512_{procver}.fits",
                bndctr=30,
                restfreq=30,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="040-WMAP_Q1",
                fwhm=0.0,
                nu_ref_t="40 GHz",
                nu_ref_p="40 GHz",
                procver=procver,
                filename=f"BP_040-WMAP_Q1_IQU_n0512_{procver}.fits",
                bndctr=40,
                restfreq=40,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="040-WMAP_Q2",
                fwhm=0.0,
                nu_ref_t="40 GHz",
                nu_ref_p="40 GHz",
                procver=procver,
                filename=f"BP_040-WMAP_Q2_IQU_n0512_{procver}.fits",
                bndctr=40,
                restfreq=40,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="060-WMAP_V1",
                fwhm=0.0,
                nu_ref_t="60 GHz",
                nu_ref_p="60 GHz",
                procver=procver,
                filename=f"BP_060-WMAP_V1_IQU_n0512_{procver}.fits",
                bndctr=60,
                restfreq=60,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="060-WMAP_V2",
                fwhm=0.0,
                nu_ref_t="60 GHz",
                nu_ref_p="60 GHz",
                procver=procver,
                filename=f"BP_060-WMAP_V2_IQU_n0512_{procver}.fits",
                bndctr=60,
                restfreq=60,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="090-WMAP_W1",
                fwhm=0.0,
                nu_ref_t="90 GHz",
                nu_ref_p="90 GHz",
                procver=procver,
                filename=f"BP_090-WMAP_W1_IQU_n0512_{procver}.fits",
                bndctr=90,
                restfreq=90,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="090-WMAP_W2",
                fwhm=0.0,
                nu_ref_t="90 GHz",
                nu_ref_p="90 GHz",
                procver=procver,
                filename=f"BP_090-WMAP_W2_IQU_n0512_{procver}.fits",
                bndctr=90,
                restfreq=90,
                bndwid=5,
            )
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="090-WMAP_W3",
                fwhm=0.0,
                nu_ref_t="90 GHz",
                nu_ref_p="90 GHz",
                procver=procver,
                filename=f"BP_090-WMAP_W3_IQU_n0512_{procver}.fits",
                bndctr=90,
                restfreq=90,
                bndwid=5,
            )

            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK", "mK2", "mK2", "mK2"],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="090-WMAP_W4",
                fwhm=0.0,
                nu_ref_t="90 GHz",
                nu_ref_p="90 GHz",
                procver=procver,
                filename=f"BP_090-WMAP_W4_IQU_n0512_{procver}.fits",
                bndctr=90,
                restfreq=90,
                bndwid=5,
            )
            
            # Full-mission 30 GHz IQU frequency map
            # BP_030_IQU_n0512_{procver}.fits
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK2", "uK2", "uK2",],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="030",
                fwhm=0.0,
                nu_ref_t="30.0 GHz",
                nu_ref_p="30.0 GHz",
                procver=procver,
                filename=f"BP_030_IQU_n0512_{procver}.fits",
                bndctr=30,
                restfreq=28.456,
                bndwid=9.899,
            )
            # Full-mission 44 GHz IQU frequency map
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK2", "uK2", "uK2",],
                nside=512,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="044",
                fwhm=0.0,
                nu_ref_t="44.0 GHz",
                nu_ref_p="44.0 GHz",
                procver=procver,
                filename=f"BP_044_IQU_n0512_{procver}.fits",
                bndctr=44,
                restfreq=44.121,
                bndwid=10.719,
            )
            # Full-mission 70 GHz IQU frequency map
            format_fits(
                chain=chain,
                extname="FREQMAP",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_RMS", "Q_RMS", "U_RMS","I_STDDEV", "Q_STDDEV", "U_STDDEV", "IQ_COV", "IU_COV", "QU_COV"],
                units=["uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK", "uK2", "uK2", "uK2",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="070",
                fwhm=0.0,
                nu_ref_t="70.0 GHz",
                nu_ref_p="70.0 GHz",
                procver=procver,
                filename=f"BP_070_IQU_n1024_{procver}.fits",
                bndctr=70,
                restfreq=70.467,
                bndwid=14.909,
            )



        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    
    """
    FOREGROUND MAPS
    """
    # Full-mission CMB IQU map
    if cmb:
        if resamp:
            if pol == "P":
                try:
                    format_fits(
                        chain,
                        extname="COMP-MAP-CMB-RESAMP-P",
                        types=["Q_MEAN","U_MEAN", "Q_STDDEV","U_STDDEV",],
                        units=["uK_cmb", "uK_cmb","uK_cmb", "uK_cmb",],
                        nside=1024,
                        burnin=burnin,
                        maxchain=maxchain,
                        polar=True,
                        component="CMB",
                        fwhm=14.0,
                        nu_ref_t="NONE",
                        nu_ref_p="NONE",
                        procver=procver,
                        filename=f"BP_cmb_resamp_QU_n1024_{procver}.fits",
                        bndctr=None,
                        restfreq=None,
                        bndwid=None,
                    )
                except Exception as e:
                    print(e)
                    click.secho("Continuing...",fg="yellow")

            else:    
                try:
                    format_fits(
                        chain,
                        extname="COMP-MAP-CMB-RESAMP-T",
                        types=["I_MEAN", "I_STDDEV",],
                        units=["uK_cmb", "uK_cmb",],
                        nside=1024,
                        burnin=burnin,
                        maxchain=maxchain,
                        polar=True,
                        component="CMB",
                        fwhm=14.0,
                        nu_ref_t="NONE",
                        nu_ref_p="NONE",
                        procver=procver,
                        filename=f"BP_cmb_resamp_I_n1024_{procver}.fits",
                        bndctr=None,
                        restfreq=None,
                        bndwid=None,
                    )
                except Exception as e:
                    print(e)
                    click.secho("Continuing...",fg="yellow")
                
        else:
            try:
                format_fits(
                    chain,
                    extname="COMP-MAP-CMB",
                    types=["I_MEAN", "Q_MEAN", "U_MEAN", "I_STDDEV", "Q_STDDEV", "U_STDDEV", "mask1", "mask2",],
                    units=["uK_cmb", "uK_cmb", "uK", "uK", "NONE", "NONE",],
                    nside=1024,
                    burnin=burnin,
                    maxchain=maxchain,
                    polar=True,
                    component="CMB",
                    fwhm=14.0,
                    nu_ref_t="NONE",
                    nu_ref_p="NONE",
                    procver=procver,
                    filename=f"BP_cmb_IQU_n1024_{procver}.fits",
                    bndctr=None,
                    restfreq=None,
                    bndwid=None,
                )
            except Exception as e:
                print(e)
                click.secho("Continuing...",fg="yellow")

    if ff:
        try:
            # Full-mission free-free I map
            format_fits(
                chain,
                extname="COMP-MAP-FREE-FREE",
                types=["I_MEAN", "I_TE_MEAN", "I_STDDEV", "I_TE_STDDEV",],
                units=["uK_RJ", "K", "uK_RJ", "K",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=False,
                component="FREE-FREE",
                fwhm=30.0,
                nu_ref_t="40.0 GHz",
                nu_ref_p="40.0 GHz",
                procver=procver,
                filename=f"BP_freefree_I_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if ame:
        try:
            # Full-mission AME I map
            format_fits(
                chain,
                extname="COMP-MAP-AME",
                types=["I_MEAN", "I_NU_P_MEAN", "I_STDDEV", "I_NU_P_STDDEV",],
                units=["uK_RJ", "GHz", "uK_RJ", "GHz",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=False,
                component="AME",
                fwhm=120.0,
                nu_ref_t="22.0 GHz",
                nu_ref_p="22.0 GHz",
                procver=procver,
                filename=f"BP_ame_I_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if synch:
        try:
            # Full-mission synchrotron IQU map
            format_fits(
                chain,
                extname="COMP-MAP-SYNCHROTRON",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_BETA_MEAN", "QU_BETA_MEAN", "I_STDDEV", "Q_STDDEV", "U_STDDEV", "P_STDDEV", "I_BETA_STDDEV", "QU_BETA_STDDEV",],
                units=["uK_RJ", "uK_RJ", "uK_RJ", "uK_RJ", "NONE", "NONE", "uK_RJ","uK_RJ","uK_RJ","uK_RJ", "NONE", "NONE",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="SYNCHROTRON",
                fwhm=60.0,  # 60.0,
                nu_ref_t="0.408 GHz",
                nu_ref_p="30.0 GHz",
                procver=procver,
                filename=f"BP_synch_IQU_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if dust:
        try:
            # Full-mission thermal dust IQU map
            format_fits(
                chain,
                extname="COMP-MAP-DUST",
                types=["I_MEAN", "Q_MEAN", "U_MEAN", "P_MEAN", "I_BETA_MEAN", "QU_BETA_MEAN", "I_T_MEAN", "QU_T_MEAN", "I_STDDEV", "Q_STDDEV", "U_STDDEV", "P_STDDEV", "I_BETA_STDDEV", "QU_BETA_STDDEV", "I_T_STDDEV", "QU_T_STDDEV",],
                units=["uK_RJ", "uK_RJ", "uK_RJ", "uK_RJ", "NONE", "NONE", "K", "K", "uK_RJ","uK_RJ","uK_RJ","uK_RJ", "NONE", "NONE", "K", "K",],
                nside=1024,
                burnin=burnin,
                maxchain=maxchain,
                polar=True,
                component="DUST",
                fwhm=10.0,  # 60.0,
                nu_ref_t="545 GHz",
                nu_ref_p="353 GHz",
                procver=procver,
                filename=f"BP_dust_IQU_n1024_{procver}.fits",
                bndctr=None,
                restfreq=None,
                bndwid=None,
            )
        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if diff:
        import healpy as hp
        try:
            if not os.path.exists(f"{procver}/diffs"):
                os.mkdir(f"{procver}/diffs")
            click.echo("Creating frequency difference maps")
            path_dx12 = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/dx12"
            path_npipe = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/npipe"
            path_BP10 = "/mn/stornext/d16/cmbco/bp/delivery/v10.00/v2"
            maps_dx12 = ["30ghz_2018_n1024_beamscaled_dip.fits","44ghz_2018_n1024_beamscaled_dip.fits","70ghz_2018_n1024_beamscaled_dip.fits"]
            maps_npipe = ["npipe6v20_030_map_uK.fits", "npipe6v20_044_map_uK.fits", "npipe6v20_070_map_uK.fits",]
            maps_BP10  = ["BP_030_IQU_n0512_v2.fits", "BP_044_IQU_n0512_v2.fits", "BP_070_IQU_n1024_v2.fits"]
            maps_BP = [f"BP_030_IQU_n0512_{procver}.fits", f"BP_044_IQU_n0512_{procver}.fits", f"BP_070_IQU_n1024_{procver}.fits",]
            beamscaling = [9.8961854E-01, 9.9757886E-01, 9.9113965E-01]
            
            for i, freq in enumerate(["030", "044", "070",]):
                map_BP    = hp.read_map(f"{procver}/{maps_BP[i]}", field=(0,1,2),   dtype=None)
                map_npipe = hp.read_map(f"{path_npipe}/{maps_npipe[i]}", field=(0,1,2),   dtype=None)
                map_dx12  = hp.read_map(f"{path_dx12}/{maps_dx12[i]}", field=(0,1,2),   dtype=None)
                map_BP10  = hp.read_map(f"{path_BP10}/{maps_BP10[i]}", field=(0,1,2),   dtype=None)
                
                #dx12 dipole values:
                # 3362.08 pm 0.99, 264.021 pm 0.011, 48.253 ± 0.005
                # 233.18308357  2226.43833645 -2508.42179665
                #dipole_dx12 = -3362.08*hp.dir2vec(264.021, 48.253, lonlat=True)

                #map_dx12  = map_dx12/beamscaling[i]
                # Smooth to 60 arcmin
                map_BP = hp.smoothing(map_BP, fwhm=arcmin2rad(60.0))
                map_npipe = hp.smoothing(map_npipe, fwhm=arcmin2rad(60.0))
                map_dx12 = hp.smoothing(map_dx12, fwhm=arcmin2rad(60.0))
                map_BP10 = hp.smoothing(map_BP10, fwhm=arcmin2rad(60.0))

                #ud_grade 30 and 44ghz
                if i<2:
                    map_npipe = hp.ud_grade(map_npipe, nside_out=512,)
                    map_dx12 = hp.ud_grade(map_dx12, nside_out=512,)
                    map_BP10 = hp.ud_grade(map_BP10, nside_out=512,)

                # Remove monopoles
                map_BP -= np.mean(map_BP,axis=1).reshape(-1,1)
                map_npipe -= np.mean(map_npipe,axis=1).reshape(-1,1)
                map_dx12 -= np.mean(map_dx12,axis=1).reshape(-1,1)
                map_BP10 -= np.mean(map_BP10,axis=1).reshape(-1,1)
                click.echo(f"creating {freq} GHz difference")
                hp.write_map(f"{procver}/diffs/BP_{freq}_diff_npipe_{procver}.fits", np.array(map_BP-map_npipe), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"], dtype=None)
                hp.write_map(f"{procver}/diffs/BP_{freq}_diff_dx12_{procver}.fits", np.array(map_BP-map_dx12), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"], dtype=None)
                hp.write_map(f"{procver}/diffs/BP_{freq}_diff_BP10_{procver}.fits", np.array(map_BP-map_BP10), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"], dtype=None)
            

            path_wmap9 = "/mn/stornext/d16/cmbco/ola/wmap/freq_maps"
            maps_wmap9  = ["wmap_iqusmap_r9_9yr_K1_v5.fits",
                           "wmap_iqusmap_r9_9yr_Ka1_v5.fits",
                           "wmap_iqusmap_r9_9yr_Q1_v5.fits", "wmap_iqusmap_r9_9yr_Q2_v5.fits",
                           "wmap_iqusmap_r9_9yr_V1_v5.fits", "wmap_iqusmap_r9_9yr_V2_v5.fits",
                           "wmap_iqusmap_r9_9yr_W1_v5.fits", "wmap_iqusmap_r9_9yr_W2_v5.fits",
                           "wmap_iqusmap_r9_9yr_W3_v5.fits", "wmap_iqusmap_r9_9yr_W4_v5.fits"]
            maps_BP     = [f"BP_023-WMAP_K_IQU_n0512_{procver}.fits",
                           f"BP_030-WMAP_Ka_IQU_n0512_{procver}.fits",
                           f"BP_040-WMAP_Q1_IQU_n0512_{procver}.fits",
                           f"BP_040-WMAP_Q2_IQU_n0512_{procver}.fits",
                           f"BP_060-WMAP_V1_IQU_n0512_{procver}.fits",
                           f"BP_060-WMAP_V2_IQU_n0512_{procver}.fits",
                           f"BP_090-WMAP_W1_IQU_n0512_{procver}.fits",
                           f"BP_090-WMAP_W2_IQU_n0512_{procver}.fits",
                           f"BP_090-WMAP_W3_IQU_n0512_{procver}.fits",
                           f"BP_090-WMAP_W4_IQU_n0512_{procver}.fits"]
                              


            # WMAP9 maps must have dipole added back in:
            # Jarosik 2011 gives (unchanged from 5-year release)
            # d = 3.355, l = 263.99, b = 48.26
            d_x = -0.233
            d_y = -2.222
            d_z = 2.504
            import healpy as hp
            x, y, z= hp.pix2vec(512, np.arange(12*512**2))
            dip = d_x*x + d_y*y + d_z*z
            for i, freq in enumerate(["023-WMAP_K", "030-WMAP_Ka",
                "040-WMAP_Q1", "040-WMAP_Q2", "060-WMAP_V1", "060-WMAP_V2",
                "090-WMAP_W1", "090-WMAP_W2", "090-WMAP_W3", "090-WMAP_W4"]):
                map_BP    = hp.read_map(f"{procver}/{maps_BP[i]}", field=(0,1,2),   dtype=None)
                map_wmap9 = hp.read_map(f"{path_wmap9}/{maps_wmap9[i]}", field=(0,1,2),   dtype=None)

                map_wmap9[0] += dip

                

                #map_dx12  = map_dx12/beamscaling[i]
                # Smooth to 60 arcmin
                map_BP = hp.smoothing(map_BP, fwhm=arcmin2rad(60.0))
                map_wmap9 = hp.smoothing(map_wmap9, fwhm=arcmin2rad(60.0))

                # Remove monopoles
                map_BP -= np.mean(map_BP,axis=1).reshape(-1,1)
                map_wmap9 -= np.mean(map_wmap9,axis=1).reshape(-1,1)
                click.echo(f"creating {freq} GHz difference")
                hp.write_map(f"{procver}/diffs/BP_{freq}_diff_wmap9_{procver}.fits", np.array(map_BP-map_wmap9), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"], dtype=None)
        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if diffcmb:
        import healpy as hp
        try:
            if not os.path.exists(f"{procver}/diffs"):
                os.mkdir(f"{procver}/diffs")
            click.echo("Creating cmb difference maps")
            path_cmblegacy = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/cmb-legacy"
            mask_ = hp.read_map("/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits",   dtype=np.bool,)
            map_BP = hp.read_map(f"{procver}/BP_cmb_IQU_n1024_{procver}.fits", field=(0,1,2),   dtype=None,)
            map_BP_masked = hp.ma(map_BP[0])
            map_BP_masked.mask = np.logical_not(mask_)
            mono, dip = hp.fit_dipole(map_BP_masked)
            nside = 1024
            ray = range(hp.nside2npix(nside))
            vecs = hp.pix2vec(nside, ray)
            dipole = np.dot(dip, vecs)
            map_BP[0] = map_BP[0] - dipole - mono
            map_BP = hp.smoothing(map_BP, fwhm=arcmin2rad(np.sqrt(60.0**2-14**2)))
            #map_BP -= np.mean(map_BP,axis=1).reshape(-1,1)
            for i, method in enumerate(["commander", "sevem", "nilc", "smica",]):

                data = f"COM_CMB_IQU-{method}_2048_R3.00_full.fits"
                click.echo(f"making difference map with {data}")
                map_cmblegacy  = hp.read_map(f"{path_cmblegacy}/{data}", field=(0,1,2),  dtype=None)
                map_cmblegacy = hp.smoothing(map_cmblegacy, fwhm=arcmin2rad(60.0))
                map_cmblegacy = hp.ud_grade(map_cmblegacy, nside_out=1024,)
                map_cmblegacy = map_cmblegacy*1e6

                # Remove monopoles
                map_cmblegacy_masked = hp.ma(map_cmblegacy[0])
                map_cmblegacy_masked.mask = np.logical_not(mask_)
                mono = hp.fit_monopole(map_cmblegacy_masked)
                click.echo(f"{method} subtracting monopole {mono}")
                map_cmblegacy[0] = map_cmblegacy[0] - mono #np.mean(map_cmblegacy,axis=1).reshape(-1,1)

                hp.write_map(f"{procver}/diffs/BP_cmb_diff_{method}_{procver}.fits", np.array(map_BP-map_cmblegacy), overwrite=True, column_names=["I_DIFF", "Q_DIFF", "U_DIFF"], dtype=None)

        except Exception as e:
            print(e)
            click.secho("Continuing...",fg="yellow")

    if goodness:
        import healpy as hp
        path_goodness = procver + "/goodness"
        Path(path_goodness).mkdir(parents=True, exist_ok=True)
        print("PATH", path_goodness)

        if len(chains) == 1:
            cmin = 1 
            cmax = None
            chdir = os.path.split(chains[0])[0].rsplit("chain_", 1)[0]
        else:
            cmin = int(os.path.split(chains[0])[0].rsplit("_c")[-1])
            cmax = int(os.path.split(chains[-1])[0].rsplit("_c")[-1])
            chdir = os.path.split(chains[0])[0].rsplit("_", 1)[0]
 
        if chisq:
            try:
                format_fits(
                    chains,
                    extname="CHISQ",
                    types=["I_MEAN", "P_MEAN",],
                    units=["NONE", "NONE",],
                    nside=16,
                    burnin=burnin,
                    maxchain=maxchain,
                    polar=True,
                    component="CHISQ",
                    fwhm=0.0,
                    nu_ref_t="NONE",
                    nu_ref_p="NONE",
                    procver=procver,
                    filename=f'goodness/BP_chisq_n16_{procver}.fits',
                    bndctr=None,
                    restfreq=None,
                    bndwid=None,
                    cmin=cmin,
                    cmax=cmax,
                    chdir=chdir,
                )
            except Exception as e:
                print(e)
                click.secho("Continuing...",fg="yellow")
                
        if res:
            click.echo("Save and format chisq map and residual maps")        
            bands = {        
                     "030": {
                            "nside": 512,
                            "fwhm": 120,
                            "sig": "IQU",
                            "fields": (0,1,2),
                            "unit": "uK",
                            "scale": 1.,
                            },
                     "044": {
                             "nside": 512,
                             "fwhm": 120,
                             "sig": "IQU",
                             "fields": (0,1,2),
                             "unit": "uK",
                             "scale": 1.,
                     },
                     "070": {
                         "nside": 1024,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "uK",
                         "scale": 1.,
                     },
                     "023-WMAP_K": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1,
                     },
                     "030-WMAP_Ka": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1,
                     },
                     "040-WMAP_Q1": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "040-WMAP_Q2": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2,),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "060-WMAP_V1": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "060-WMAP_V2": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "090-WMAP_W1": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "090-WMAP_W2": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "090-WMAP_W3": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "090-WMAP_W4": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "IQU",
                         "fields": (0,1,2),
                         "unit": "mK",
                         "scale": 1.,
                     },
                     "0.4-Haslam": {
                         "nside": 512,
                         "fwhm": 120,
                         "sig": "I",
                         "fields": (0,),
                         "unit": "uK",
                         "scale": 1.,
                     },
                     "857": {
                         "nside": 1024,
                         "fwhm": 120,
                         "sig": "I",
                         "fields": (0,),
                         "unit": "uK",
                         "scale": 1.,
                     },
                    "033-WMAP_Ka_P": {
                         "nside": 16,
                         "fwhm": 0,
                         "sig": "QU",
                         "fields": (1,2),
                         "unit": "uK",
                         "scale": 1e3,
                     },
                     "041-WMAP_Q_P": {
                         "nside": 16,
                         "fwhm": 0,
                         "sig": "QU",
                         "fields": (1,2),
                         "unit": "uK",
                         "scale": 1e3,
                     },
                     "061-WMAP_V_P": {
                         "nside": 16,
                         "fwhm": 0,
                         "sig": "QU",
                         "fields": (1,2),
                         "unit": "uK",
                         "scale": 1e3,
                     },
                     "353": {
                         "nside": 1024,
                         "fwhm": 120,
                         "sig": "QU",
                         "fields": (1,2),
                         "unit": "uK",
                         "scale": 1.,
                     },
                 }
            
            
            for label, b in bands.items():
                
                types=[]
                units=[]
                for l in b["sig"]:
                    types.append(f'{l}_MEAN')
                    units.append(b["unit"])
                for l in b["sig"]:
                    types.append(f'{l}_STDDEV')
                    units.append(b["unit"])
                try:
                    format_fits(
                        chains,
                        extname="FREQBAND_RES",
                        types=types,
                        units=units, 
                        nside=b["nside"],
                        burnin=burnin,
                        maxchain=maxchain,
                        polar=True,
                        component=label,
                        fwhm=b["fwhm"],
                        nu_ref_t="NONE",
                        nu_ref_p="NONE",
                        procver=procver,
                        filename=f'goodness/BP_res_{label}_{b["sig"]}_n{b["nside"]}_{b["fwhm"]}arcmin_{b["unit"]}_{procver}.fits',
                        bndctr=None,
                        restfreq=None,
                        bndwid=None,
                        cmin=cmin,
                        cmax=cmax,
                        chdir=chdir,
                        fields=b["fields"],
                        scale=b["scale"],
                    )
                except Exception as e:
                    print(e)
                    click.secho("Continuing...",fg="yellow")

            
    """ As implemented by Simone
    """
    if br and resamp:
        # Gaussianized TT Blackwell-Rao input file
        click.echo("{:-^50}".format("CMB GBR"))
        ctx.invoke(sigma_l2fits, filename=resamp, nchains=1, burnin=burnin, path="cmb/sigma_l", outname=f"{procver}/BP_cmb_GBRlike_{procver}.fits", save=True,)

    """
    TODO Generalize this so that they can be generated by Elina and Anna-Stiina
    """
    # Full-mission 30 GHz IQU beam symmetrized frequency map
    # BP_030_IQUdeconv_n0512_{procver}.fits
    # Full-mission 44 GHz IQU beam symmetrized frequency map
    # BP_044_IQUdeconv_n0512_{procver}.fits
    # Full-mission 70 GHz IQU beam symmetrized frequency map
    # BP_070_IQUdeconv_n1024_{procver}.fits

    """ Both sigma_l's and Dl's re in the h5. (Which one do we use?)
    """
    # CMB TT, TE, EE power spectrum
    # BP_cmb_{procver}.txt

    """ Just get this from somewhere
    """
    # Best-fit LCDM CMB TT, TE, EE power spectrum
    # BP_cmb_bfLCDM_{procver}.txt

    if plot:
        os.chdir(procver)
        ctx.invoke(plotrelease, procver=procver, all_=True)
