import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from .plottools import load_cmap, fmt, autoparams, apply_logscale, set_style, make_fig


def gnomplot(
    map_,
    lon,
    lat,
    auto=None,
    sig=0,
    size=20,
    vmin=None,
    vmax=None,
    rng=None,
    title=None,
    ltitle=None,
    unit=None,
    cmap="planck",
    graticule=False,
    logscale=False,
    colorbar=True,
    fwhm=0.0,
    remove_dip=False,
    remove_mono=False,
    fontsize=11,
    figsize=(5, 6),
    darkmode=False,
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
):
    """
    Gnomonic view plotting.
    """
    # Set plotting rcParams
    set_style(darkmode)

    xsize = 5000
    reso = size * 60 / xsize
    if isinstance(map_, str):
        x = hp.read_map(map_, field=sig, verbose=False, dtype=None)
    else:
        if map_.ndim > 1:
            x = map_[sig]
        else:
            x = map_

    nside = hp.get_nside(x)

    if float(fwhm) > 0:
        x = hp.smoothing(x, fwhm=fwhm * (2 * np.pi) / 21600, lmax=3 * nside,)
    if remove_dip:
        x = hp.remove_dipole(x, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        x = hp.remove_monopole(x, gal_cut=30, copy=True, verbose=True)

    proj = hp.projector.GnomonicProj(
        rot=[lon, lat, 0.0], coord="G", xsize=xsize, ysize=xsize, reso=reso
    )
    reproj_im = proj.projmap(x, vec2pix_func=partial(hp.vec2pix, nside))

    if rng is not None:
        vmin = -rng
        vmax = rng

    params = autoparams(auto, sig, title, ltitle, unit, None, logscale, cmap)
    vmin, vmax = (params["ticks"][0], params["ticks"][-1])
    if params["logscale"]:
        x, (vmin, vmax) = apply_logscale(x, [vmin, vmax], linthresh=1)

    cmap = load_cmap(params["cmap"], params["logscale"])

    # Format for latex
    for i in ["title", "unit", "left_title"]:
        if params[i] and params[i] != "":
            params[i] = r"$" + params[i] + "$"

    fig, ax = make_fig(figsize, fignum, hold, subplot, reuse_axes,)
    image = plt.imshow(
        reproj_im,
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.xticks([])
    plt.yticks([])
    plt.text(
        0.95,
        0.95,
        params["title"],
        color="black",
        va="top",
        ha="right",
        transform=ax.transAxes,
        fontsize=fontsize + 4,
    )
    plt.text(
        0.05,
        0.95,
        params["left_title"],
        color="black",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize + 4,
    )
    if colorbar:
        # colorbar
        from matplotlib.ticker import FuncFormatter

        cb = plt.colorbar(
            image,
            orientation="horizontal",
            shrink=0.5,
            pad=0.03,
            format=FuncFormatter(fmt),
        )
        cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize)
        cb.ax.xaxis.set_label_text(params["unit"])
        cb.ax.xaxis.label.set_size(fontsize)

    if graticule:
        hp.graticule()
