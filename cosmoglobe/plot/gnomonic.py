import healpy as hp
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
    cbar_pad=0.04,
    cbar_shrink=0.7,
    fwhm=0.0 * u.arcmin,
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
    xsize = 1000
    reso = size * 60 / xsize

    # Get data
    m, comp, freq, nside  = get_data(input, sig, comp, freq, fwhm, nside=nside, sample=sample)
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    proj = hp.projector.GnomonicProj(
        rot=[lon, lat, 0.0], coord="G", xsize=xsize, ysize=xsize, reso=reso
    )
    reproj_im = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))

    # Pass all your arguments in, return parsed plotting parameters
    params = get_params(
        data=reproj_im, 
        comp=comp,
        sig=sig,
        right_label=right_label,
        left_label=left_label,
        unit=unit,
        ticks=ticks,
        min=vmin,
        max=vmax,
        rng=rng,
        norm=norm,
        cmap=cmap,
        freq_ref=freq,
        width=figsize[0],
        nside=nside
    )

    # Semi-log normalization
    if params["norm"] == "log":
        params["data"], params["ticks"] = apply_logscale(params["data"], params["ticks"], linthresh=1)

    cmap = load_cmap(params["cmap"])

    fig, ax = make_fig(
        figsize,
        fignum,
        hold,
        subplot,
        reuse_axes
    )
    image = plt.imshow(
        params["data"],
        origin="lower",
        interpolation="nearest",
        vmin=params["ticks"][0],
        vmax=params["ticks"][-1],
        cmap=cmap
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
        fontsize=fontsize["title"]
    )
    plt.text(
        0.05,
        0.95,
        params["left_label"],
        color="black",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize["title"]
    )

    if cbar:
        apply_colorbar(
            plt.gcf(),
            plt.gca(),
            image,
            params["ticks"],
            params["ticklabels"],
            params["unit"],
            fontsize=fontsize,
            linthresh=1,
            norm=params["norm"],
            cbar_pad=cbar_pad,
            cbar_shrink=cbar_shrink
        )

    if graticule:
        hp.graticule()

    return image, params
