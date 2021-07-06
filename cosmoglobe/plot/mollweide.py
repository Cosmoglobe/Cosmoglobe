import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .. import data as data_dir
from .plottools import *

hp.disable_warnings()
# Fix for macos openMP duplicate bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
"""
HVA VIL JEG AT DENNE SKAL GJØRE NÅ SOM HOVEDFFUNKSJONALITETEN FFINNES I HP:

fjerne mono-dipol
smoothe
udgrade
maskere
sette left og right title
lognorm
autoparams
subplots

Skal jeg heller lage en funksjon som returnerer riktige parametre til projview?

skriv en plottingfunksjon i cgp som er en wrapper på projview
hvis man passer inn return_plotparams får man data + projview kwargs
"""

def mollplot(
    map_,
    sig=0,
    comp=None,
    ticks=None,
    colorbar=True,
    unit=None,
    coord=None,
    graticule=False,
    projection_type="mollweide",
    fwhm=0.0,
    nside=None,
    mask=None,
    logscale=None,
    remove_dip=False,
    remove_mono=False,
    cmap=None,
    title=None,
    ltitle=None,
    figsize=(6, 3),
    darkmode=False,
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
    verbose=False,
    **kwargs,
):
    """
    Plots a fits file, a h5 data structure or a data array in Mollview with
    pretty formatting. Option of autodecting component base on file-string with
    json look-up table.

    Parameters
    ----------
    map_ : cosmoglobe map object
        cosmoglobe map object with I (optional Q and U)
    sig : list of int or string ["I", "Q", "U"]
        Signal indices to be plotted. 0, 1, 2 interprated as IQU.
        Supports multiple, such that -sig 1 -sig 2.
        default = [0,]
    comp : String
        Component label for automatic identification of plotting
        parameters based on information from autoparams.json default = None
    ticks : list or str
        Min and max value for data. If None, uses 97.5th percentile.
        default = None
    colorbar : bool
        Adds a colorbar, and "cb" to output filename.
        default = False
    graticule : bool
        Adds graticule to figure.
        default = False
    fwhm : float
        Optional map smoothing. FWHM of gaussian smoothing in arcmin.
        default = 0.0
    mask : array of healpix mask
        Apply a mask file to data
        default = None
    mdmask : str or array
        Mask for mono-dipole removal. If "comp", uses 30 degree cutoff.
        Removes both by default. Toggle off dipole or monopole removal with
        remove_dipole or remove_monopole arguments.
        default = None
    remove_dip : bool
        If mdmask is specified, fits and removes a dipole.
        default = True
    remove_mono : bool
        If mdmask is specified, fits and removes a monopole.
        default = True
    logscale : str
        Normalizes data using a semi-logscale linear between -1 and 1.
        Autodetector uses this sometimes, you will be warned.
        default = None
    size : str
        Size of output maps. 1/3, 1/2 and full page width (8.8/12/18cm) [ex. x,
        s, m or l, or ex. slm for all]" default = "m"
    darkmode : bool
        Plots all outlines in white for dark backgrounds, and adds "dark" in
        filename.
        default = False
    cmap : str
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default = None
    title : str
        Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    ltitle : str
        Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    fontsize : int
        Fontsize
        default = 11
    fignum : int or None, optional
      The figure number to use. Default: create a new figure
    hold : bool, optional
      If True, replace the current Axes by a MollweideAxes.
      use this if you want to have multiple maps on the same
      figure. Default: False
    sub : int, scalar or sequence, optional
      Use only a zone of the current figure (same syntax as subplot).
      Default: None
    reuse_axes : bool, optional
      If True, reuse the current Axes (should be a MollweideAxes). This is
      useful if you want to overplot with a partially transparent colormap,
      such as for plotting a line integral convolution. Default: False
    kwargs : keywords
      Additional arguments for projview
    """
    # Pick sizes from size dictionary for page width plots
    if isinstance(figsize, str):
        width = {
            "x": 2.75,
            "s": 3.5,
            "m": 4.7,
            "l": 7,
        }[figsize]
    else:
        width, height = figsize
    if colorbar:
        height *= 1.275  # Size correction with cbar
    #override_plot_properties["figure_width"] = width

    set_style(darkmode)
    """
    # Make figure
    fig, ax = make_fig(
        (width, height),
        fignum,
        hold,
        subplot,
        reuse_axes, 
        projection="mollweide",
    )
    """

    # Translate sig to correct format
    stokes = ["I", "Q", "U",]
    if isinstance(sig, str):
        sig = stokes.index(sig)

    if comp is not None:
        parse = comp.split()
        comp = parse[0]
        specparam = parse[-1]

    params = autoparams(comp, sig, title, ltitle, unit, ticks, logscale, cmap)

    if isinstance(map_, str):
        map_ = hp.read_map(map_, field=sig)

    if isinstance(map_, Model):
        # If it is a model, get a bunch of stuff from model
        if freq is not None:
            map_=map_.freq_ref
        else:
            map_=map_.comp.amp.

    m = map_[sig] if map_.ndim > 1 else map_

    # Mask map
    if mask is not None:
        m = hp.ma(m)
        m.mask = np.logical_not(mask)

    # udgrade map
    if nside != None and nside != hp.get_nside(m):
        m = hp.ud_grade(m, nside)
    else:
        nside = hp.get_nside(m)

    # Smooth map
    if fwhm > 0.0:
        m = hp.smoothing(m, fwhm)

    if verbose:
        print(f"Plotting {comp}, title {title}, nside {nside}")
    
    # Remove mono/dipole
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    # Ticks and ticklabels
    ticks = params["ticks"]
    if ticks == None:
        ticks = get_percentile(m, 97.5)

    #### Logscale ####
    ticklabels = [fmt(i, 1) for i in ticks]
    if params["logscale"]:
        m, ticks = apply_logscale(m, ticks, linthresh=1)

    #### Color map #####
    cmap = load_cmap(params["cmap"], params["logscale"])

    if verbose:
        # Terminal ouput
        print(f"FWHM: {fwhm}")
        print(f"nside: {nside}")
        print(f"Ticks: {ticklabels}")
        print(f'Unit: {params["unit"]}')
        print(f'Title: {params["title"]}')

    for i in ["title", "unit", "left_title"]:
        if params[i] and params[i] != "":
            params[i] = r"$" + params[i] + "$"

    ret = hp.newvisufunc.projview(m, 
            min=ticks[0], 
            max=ticks[-1],
            cbar=False,
            cmap=cmap,
            projection_type=projection_type,
            graticule=graticule,
            coord=coord, 
            #override_plot_properties=override_plot_properties
            **kwargs
            )
    
    # Remove colorbar because of healpy bug
    plt.gca().collections[-1].colorbar.remove()
    if colorbar:
        apply_colorbar(
            plt.gcf(), plt.gca(), ret, ticks, ticklabels, params["unit"], linthresh=1, logscale=params["logscale"])
    #### Right Title ####
    plt.text(4.5, 1.1, params["title"], ha="center", va="center",)
    #### Left Title (stokes parameter label by default) ####
    plt.text(-4.5, 1.1, params["left_title"], ha="center", va="center",)

def apply_colorbar(fig, ax, image, ticks, ticklabels, unit, linthresh, logscale=False)s:
    """
    This function applies a colorbar to the figure and formats the ticks.
    """
    from matplotlib.ticker import FuncFormatter

    cb = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        shrink=0.4,
        pad=0.04,
        ticks=ticks,
        format=FuncFormatter(fmt),
    )
    cb.ax.set_xticklabels(ticklabels)
    cb.ax.xaxis.set_label_text(unit)
    #cb.ax.xaxis.label.set_size(fontsize)
    if logscale:
        linticks = np.linspace(-1, 1, 3) * linthresh
        logmin = np.round(ticks[0])
        logmax = np.round(ticks[-1])

        logticks_min = -(10 ** np.arange(0, abs(logmin) + 1))
        logticks_max = 10 ** np.arange(0, logmax + 1)
        ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))
        # cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks

        logticks = symlog(ticks_, linthresh)
        logticks = [x for x in logticks if x not in ticks]
        cb.set_ticks(np.concatenate((ticks, logticks)))  # Set major ticks
        cb.ax.set_xticklabels(ticklabels + [""] * len(logticks))

        minorticks = np.linspace(-linthresh, linthresh, 5)
        minorticks2 = np.arange(2, 10) * linthresh

        for i in range(len(logticks_min)):
            minorticks = np.concatenate((-(10 ** i) * minorticks2, minorticks))
        for i in range(len(logticks_max)):
            minorticks = np.concatenate((minorticks, 10 ** i * minorticks2))

        minorticks = symlog(minorticks, linthresh)
        minorticks = minorticks[(minorticks >= ticks[0]) & (minorticks <= ticks[-1])]
        cb.ax.xaxis.set_ticks(minorticks, minor=True)

    cb.ax.tick_params(
        which="both",
        axis="x",
        direction="in",
        #labelsize=fontsize - 2,
    )
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb
