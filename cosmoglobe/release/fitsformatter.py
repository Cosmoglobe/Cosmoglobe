import time
import sys
import os
import numpy as np
import healpy as hp
from cosmoglobe.release.tools import *
import cosmoglobe as cg


def format_fits(chain, extname, types, units, nside, burnin, maxchain, polar, component, fwhm, nu_ref_t, nu_ref_p, procver, filename, bndctr, restfreq, bndwid, cmin=1, cmax=None, chdir=None, fields=None, scale=1., coadd=False):
    print()
    print("{:#^80}".format(""))
    print("{:#^80}".format(f" Formatting and outputting {filename} "))
    print("{:#^80}".format(""))
    
    if coadd:
        header = get_header(extname, types, units, nside, polar, component[0], fwhm, nu_ref_t, nu_ref_p, procver, filename, bndctr, restfreq, bndwid,)
    else:
        header = get_header(extname, types, units, nside, polar, component, fwhm, nu_ref_t, nu_ref_p, procver, filename, bndctr, restfreq, bndwid,)
    dset = get_data(chain, extname, component, burnin, maxchain, fwhm, nside, types, cmin, cmax, chdir, fields, scale, polar, coadd,)

    print(f"{procver}/{filename}", dset.shape)
    hp.write_map(f"{procver}/{filename}", dset, column_names=types, column_units=units, coord="G", overwrite=True, extra_header=header, dtype=None)


def get_data(chain, extname, component, burnin, maxchain, fwhm, nside, types, cmin, cmax, chdir, fields=None, scale=1.0, polar=True, coadd=False,):



    if extname.endswith("CMB"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="cmb/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="cmb/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        # Masks
        mask1 = np.zeros((hp.nside2npix(nside)))

        mask2 = np.zeros((hp.nside2npix(nside)))

        dset = np.zeros((len(types), hp.nside2npix(nside)))
        dset[0] = amp_mean[0, :]
        dset[1] = amp_mean[1, :]
        dset[2] = amp_mean[2, :]
        
        dset[3] = amp_stddev[0, :]
        dset[4] = amp_stddev[1, :]
        dset[5] = amp_stddev[2, :]
        
        dset[6] = mask1
        dset[7] = mask2

    if extname.endswith("RESAMP-T"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="cmb/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="cmb/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))
        dset[0] = amp_mean
        dset[1] = amp_stddev
    elif extname.endswith("RESAMP-P"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="cmb_lowl/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="cmb_lowl/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))
        dset[0] = amp_mean[0,:]
        dset[1] = amp_mean[1,:]
        dset[2] = amp_stddev[0,:]
        dset[3] = amp_stddev[1,:]
    elif extname.endswith("SYNCHROTRON"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="synch/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        beta_mean = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="synch/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)
        beta_stddev = h5handler(input=chain, dataset="synch/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean[0, :]
        dset[1] = amp_mean[1, :]
        dset[2] = amp_mean[2, :]
        dset[3] = np.sqrt(amp_mean[1, :]**2 + amp_mean[2, :]**2)

        dset[4] = beta_mean[0, :]
        dset[5] = beta_mean[1, :]

        dset[6] = amp_stddev[0, :]
        dset[7] = amp_stddev[1, :]
        dset[8] = amp_stddev[2, :]
        dset[9] = np.sqrt(amp_stddev[1, :]**2 + amp_stddev[2, :]**2)

        dset[10] = beta_stddev[0, :]
        dset[11] = beta_stddev[1, :]

    elif extname.endswith("DUST"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="dust/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        beta_mean = h5handler(input=chain, dataset="dust/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)
        assert amp_mean.shape == beta_mean.shape, f"SED parameters have different nside"
        T_mean = h5handler(input=chain, dataset="dust/T_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="dust/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)
        beta_stddev = h5handler(input=chain, dataset="dust/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)
        T_stddev = h5handler(input=chain, dataset="dust/T_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)



        dset = np.zeros((len(types), hp.nside2npix(nside)))

        if len(types) == 6:

            dset[0] = amp_mean

            dset[1] = beta_mean

            dset[2] = T_mean

            dset[3] = amp_stddev

            dset[4] = beta_stddev

            dset[5] = T_stddev
        else:

            dset[0] = amp_mean[0, :]
            dset[1] = amp_mean[1, :]
            dset[2] = amp_mean[2, :]
            dset[3] = np.sqrt(amp_mean[1, :]**2 + amp_mean[2, :]**2)

            dset[4] = beta_mean[0, :]
            dset[5] = beta_mean[1, :]

            dset[6] = T_mean[0, :]
            dset[7] = T_mean[1, :]

            dset[8] = amp_stddev[0, :]
            dset[9] = amp_stddev[1, :]
            dset[10] = amp_stddev[2, :]
            dset[11] = np.sqrt(amp_stddev[1, :]**2 + amp_stddev[2, :]**2)

            dset[12] = beta_stddev[0, :]
            dset[13] = beta_stddev[1, :]

            dset[14] = T_stddev[0, :]
            dset[15] = T_stddev[1, :]

    elif extname.endswith("FREE-FREE"):
        # Mean data
        amp_mean = h5handler(input=chain, dataset="ff/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        Te_mean = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)

        # stddev data
        amp_stddev = h5handler(input=chain, dataset="ff/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)
        Te_stddev = h5handler(input=chain, dataset="ff/Te_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = Te_mean

        dset[2] = amp_stddev
        dset[3] = Te_stddev

    elif extname.endswith("AME"):
        # Mean/std amplitude 
        amp_mean = h5handler(input=chain, dataset="ame/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        amp_stddev = h5handler(input=chain, dataset="ame/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        # Mean/std spectral parameters
        c = cg.Chain(chain)
        comp_type = c.parameters['ame']['type']
        if comp_type == 'exponential':
            sed_mean = h5handler(input=chain, dataset="ame/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)
            sed_stddev = h5handler(input=chain, dataset="ame/beta_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)
        elif comp_type == 'spindust2':
            sed_mean = h5handler(input=chain, dataset="ame/nu_p_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.mean,)
            sed_stddev = h5handler(input=chain, dataset="ame/nu_p_map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=0.0, nside=nside, command=np.std,)
        else:
            print(f'Component type {comp_type} not supported')

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = sed_mean

        dset[2] = amp_stddev
        dset[3] = sed_stddev

    elif extname.endswith("CII-line"):
        # Mean/std amplitude 
        amp_mean = h5handler(input=chain, dataset="dust_cii/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        amp_stddev = h5handler(input=chain, dataset="dust_cii/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = amp_stddev

    elif extname.endswith("stars"):
        # Mean/std amplitude 
        # amp_mean = h5handler(input=chain, dataset="dust_cii/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        # amp_stddev = h5handler(input=chain, dataset="dust_cii/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        amp_mean = fits_handler(input="stars_01a_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, drop_missing=True, pixweight=None, command=np.mean, lowmem=False, fields=fields, write=False, zerospin=False)
        amp_stddev = fits_handler(input="stars_01a_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, drop_missing=True, pixweight=None, command=np.std, lowmem=False, fields=fields, write=False, zerospin=False)
        #amp_mean = fits_handler(input="chisq_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=False, drop_missing=True, pixweight=None, command=np.mean, lowmem=False, write=False)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = amp_stddev

    elif extname.endswith("PAH"):
        # Mean/std amplitude 
        amp_mean = h5handler(input=chain, dataset="hotPAH/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        amp_stddev = h5handler(input=chain, dataset="hotPAH/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = amp_stddev
    elif extname.endswith("CO_tot"):
        # Mean/std amplitude 
        amp_mean = h5handler(input=chain, dataset="co_tot/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean,)
        amp_stddev = h5handler(input=chain, dataset="co_tot/amp_alm", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.std,)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean
        dset[1] = amp_stddev

    elif extname.endswith("FREQMAP"):
        if polar:
            zerospin=False
        else:
            zerospin=True
        # Mean data
        if coadd:
            datasets = [f"tod/{comp}/map" for comp in component]
            amp_mean = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean, zerospin=zerospin, coadd=coadd, )

            if polar:
                amp_covar = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=120., nside=nside, command=np.cov, remove_mono=True, zerospin=zerospin, coadd=coadd,)
            else:
                amp_stddev = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=120., nside=nside, command=np.std, remove_mono=True, zerospin=zerospin, coadd=coadd,)

            datasets = [f"tod/{comp}/rms" for comp in component]
            amp_rms  = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean, zerospin=zerospin, coadd=coadd,)

        else:
            amp_mean = h5handler(input=chain, dataset=f"tod/{component}/map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean, zerospin=zerospin,)
            amp_rms  = h5handler(input=chain, dataset=f"tod/{component}/rms", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean, zerospin=zerospin,)
            if polar:
                amp_covar = h5handler(input=chain, dataset=f"tod/{component}/map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=120., nside=nside, command=np.cov, remove_mono=True, zerospin=zerospin,)
            else:
                amp_stddev = h5handler(input=chain, dataset=f"tod/{component}/map", min=burnin, max=None, maxchain=maxchain, output="map", fwhm=120., nside=nside, command=np.std, remove_mono=True, zerospin=zerospin,)

        # Masks

        dset = np.zeros((len(types), hp.nside2npix(nside)))


        if polar:
            dset[0] = amp_mean[0, :]
            dset[1] = amp_mean[1, :]
            dset[2] = amp_mean[2, :]

            dset[3] = amp_rms[0, :]**0.5
            dset[4] = amp_rms[1, :]**0.5
            dset[5] = amp_rms[2, :]**0.5
            dset[6] = amp_rms[3, :]

            dset[7] = amp_covar[0, 0, :]**0.5
            dset[8] = amp_covar[1, 1, :]**0.5
            dset[9] = amp_covar[2, 2, :]**0.5
            dset[10] = amp_covar[1, 2, :]
        else:
            dset[0] = amp_mean
            dset[1] = amp_rms
            dset[2] = amp_stddev


    elif extname.endswith("RES"):
        N = len(types)
        if polar:
            zerospin=False
        else:
            zerospin=True
        if coadd:
        #    datasets = [f"tod/{comp}/map" for comp in component]
        #    amp_mean = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=fwhm, nside=nside, command=np.mean, zerospin=zerospin, coadd=coadd, )

        #    amp_stddev = h5handler(input=chain, dataset=datasets, min=burnin, max=None, maxchain=maxchain, output="map", fwhm=120., nside=nside, command=np.std, remove_mono=True, zerospin=zerospin, coadd=coadd,)
            # component is a list that looks like
            # 10a, 10b, 10

            inputs = [f"res_{c}_c0001_k000001.fits" for c in component]
            rms_maps = [f"tod_{c}_rms_c0001_k000001.fits" for c in component[:-1]]
            amp_mean = fits_handler(input=inputs, min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=zerospin, drop_missing=True, pixweight=None, command=np.mean, lowmem=False, fields=fields, write=False, coadd=True, rms_maps=rms_maps)
            amp_stddev = fits_handler(input=inputs, min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=zerospin, drop_missing=True, pixweight=None, command=np.std, lowmem=False, fields=fields, write=False, coadd=True, rms_maps=rms_maps)

        else:
            amp_mean = fits_handler(input=f"res_{component}_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=zerospin, drop_missing=True, pixweight=None, command=np.mean, lowmem=False, fields=fields, write=False)
            amp_stddev = fits_handler(input=f"res_{component}_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=zerospin, drop_missing=True, pixweight=None, command=np.std, lowmem=False, fields=fields, write=False)
        dset = np.zeros((N, hp.nside2npix(nside)))
        if len(fields)>1:
            dset[:N//2] = amp_mean[fields, :]*scale
            dset[N//2:] = amp_stddev[fields, :]*scale
        else:
            dset[0] = amp_mean*scale
            dset[1] = amp_stddev*scale

    elif extname.endswith("CHISQ"):
        
        amp_mean = fits_handler(input="chisq_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=False, drop_missing=True, pixweight=None, command=np.mean, lowmem=False, write=False)
        #amp_stddev = fits_handler(input="chisq_c0001_k000001.fits", min=burnin, max=None, minchain=cmin, maxchain=cmax, chdir=chdir, output="map", fwhm=fwhm, nside=nside, zerospin=False, drop_missing=True, pixweight=None, command=np.std, lowmem=False, write=False)

        dset = np.zeros((len(types), hp.nside2npix(nside)))

        dset[0] = amp_mean[0, :]
        dset[1] = amp_mean[1, :]+amp_mean[2, :]
    else:
        print(f"Have not set up case for {extname}")

    #print(f"Shape of dset {dset.shape}")
    return dset

def get_header(extname, types, units, nside, polar, component, fwhm, nu_ref_t, nu_ref_p, procver, filename, bndctr, restfreq, bndwid,):
    stamp = f'Written {time.strftime("%c")}'

    header = []
    header.append(("DATE", stamp, "Time and date of creation.",))
    header.append(("PIXTYPE", "HEALPIX", "HEALPIX pixelisation.",))
    header.append(("COORDSYS", "GALACTIC"))
    header.append(("POLAR", polar))
    header.append(("BAD_DATA", hp.UNSEEN, "HEALPIX UNSEEN value.",))
    header.append(("METHOD", "COMMANDER", "COMMANDER sampling framework",))
    header.append(("AST-COMP", component))
    if extname == "FREQMAP":
        header.append(("FREQ", nu_ref_t))
    else:
        header.append(("FWHM", fwhm))
        header.append(("NU_REF_T", nu_ref_t))
        header.append(("NU_REF_P", nu_ref_p))

    header.append(("PROCVER", procver, "Release version"))
    header.append(("FILENAME", filename))
    if extname == "FREQMAP":
        # TODO are these correct?
        header.append(("BNDCTR", bndctr, "Formal Band Center",))
        header.append(("RESTFREQ", restfreq, "Effective Central Frequency",))
        header.append(("BNDWID", bndwid, "Effective Bandwidth",))
    return header
