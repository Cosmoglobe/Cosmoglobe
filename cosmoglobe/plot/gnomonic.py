import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from .plottools import *

@u.quantity_input(freq=u.Hz, fwhm=(u.arcmin, u.rad, u.deg))
def gnom(
    input,
    lon=0,
    lat=0,
    comp=None,
    sig=0,
    freq=None,
    nside=None,
    size=20,
    sample=-1,
    vmin=None,
    vmax=None,
    ticks=None,
    rng=None,
    left_label=None,
    right_label=None,
    unit=None,
    cmap=None,
    graticule=False,
    norm=None,
    cbar=True,
    fwhm=0.0*u.arcmin,
    remove_dip=False,
    remove_mono=False,
    fontsize=None,
    figsize=(3, 4),
    darkmode=False,
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
):
    """
    Gnomonic view plotting.
    """
    if not fontsize:
        fontsize = {
            "xlabel": 11,
            "ylabel": 11,
            "xtick_label": 8,
            "ytick_label": 8,
            "title": 12,
            "cbar_label": 11,
            "cbar_tick_label": 9,
            "left_label": 11,
            "right_label": 11,
        }

    # Set plotting rcParams
    set_style(darkmode)
    xsize = 5000
    reso = size * 60 / xsize

    # Get data
    m = get_data(input, sig, comp, freq, fwhm, nside=nside, sample=sample)

    nside = hp.get_nside(m)

    if float(fwhm.value) > 0:
        m = hp.smoothing(
            m,
            fwhm=fwhm.to(u.rad).value,
            lmax=3 * nside,
        )
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    proj = hp.projector.GnomonicProj(
        rot=[lon, lat, 0.0], coord="G", xsize=xsize, ysize=xsize, reso=reso
    )
    reproj_im = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))

    # Fetching autoset parameters
    params = autoparams(
        comp, sig, right_label, left_label, unit, ticks, vmin, vmax, norm, cmap, freq
    )


    # Ticks and ticklabels
    ticks = params["ticks"]
    if ticks == "auto" or params["norm"] == "hist":
        ticks = get_percentile(reproj_im, 97.5)
    elif None in ticks:
        pmin, pmax = get_percentile(reproj_im, 97.5)
        if ticks[0] is None:
            ticks[0] = pmin
        if ticks[-1] is None:
            ticks[-1] = pmax

    # Update parameter dictionary
    params["ticks"] = ticks

    # Create ticklabels from final ticks
    ticklabels = [fmt(i, 1) for i in ticks]
    # Semi-log normalization
    if params["norm"] == "log":
        reproj_im, ticks = apply_logscale(reproj_im, ticks, linthresh=1)

    cmap = load_cmap(params["cmap"])

    fig, ax = make_fig(
        figsize,
        fignum,
        hold,
        subplot,
        reuse_axes,
    )
    image = plt.imshow(
        reproj_im,
        origin="lower",
        interpolation="nearest",
        vmin=ticks[0],
        vmax=ticks[-1],
        cmap=cmap,
    )
    plt.xticks([])
    plt.yticks([])
    plt.text(
        0.95,
        0.95,
        params["right_label"],
        color="black",
        va="top",
        ha="right",
        transform=ax.transAxes,
        fontsize=fontsize["title"],
    )
    plt.text(
        0.05,
        0.95,
        params["left_label"],
        color="black",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize["title"],
    )
    
    if cbar:
        apply_colorbar(
            plt.gcf(),
            plt.gca(),
            image,
            ticks,
            ticklabels,
            params["unit"],
            fontsize=fontsize,
            linthresh=1,
            norm=params["norm"],
            shrink=0.7,
        )

    if graticule:
        hp.graticule()
    
    return image, params