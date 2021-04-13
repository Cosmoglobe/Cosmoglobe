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

def mollplot(
    map_,
    sig=0,
    auto=None,
    ticks=None,
    colorbar=True,
    unit=None,
    graticule=False,
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
    fontsize=11,
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
    verbose=False,
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
    auto : String
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
        Mask for mono-dipole removal. If "auto", uses 30 degree cutoff.
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
    """
    # Pick sizes from size dictionary for page width plots
    if isinstance(figsize, str):
        width = {"x": 2.75, "s": 3.5, "m": 4.7, "l": 7,}[figsize]
        height = width / 2.0
    else:
        width, height = figsize
    if colorbar: height*=1.275  # Size correction with cbar

    fig, ax = make_fig((width,height),fignum, hold, subplot, reuse_axes, projection='mollweide')

    # Translate sig to correct format
    stokes=["I", "Q", "U"]
    if isinstance(sig, str): sig=stokes.index(sig)

    params = autoparams(auto, sig, title, ltitle, unit, ticks, logscale, cmap)

    # Set plotting rcParams
    set_style(darkmode)
    if isinstance(map_, str):
        map_=hp.read_map(map_, field=sig)
    m = map_[sig] if map_.ndim > 1 else map_

    # Mask map
    if mask is not None: 
        m = hp.ma(m)    
        m.mask = np.logical_not(mask)

    # udgrade map
    if (nside != None and nside!=hp.get_nside(m)): 
        m = hp.ud_grade(m, nside)
    else:
        nside = hp.get_nside(m)

    # Smooth map
    if fwhm>0.0: 
        m = hp.smoothing(m, fwhm)

    if verbose: print(f"Plotting {auto}, title {title}, nside {nside}")

    # Projection arrays
    grid_pix, lon, lat = project_map(nside, xsize=2000, ysize=1000,)
    print(m)
    # Remove mono/dipole
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    # Put signal into map object

    grid_mask = m.mask[grid_pix] if mask else None
    grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)

    # Ticks and ticklabels
    ticks = params["ticks"]
    if ticks==None: ticks = get_percentile(m, 97.5)

    #### Logscale ####
    ticklabels = [fmt(i, 1) for i in ticks]
    if params["logscale"]: 
        m, ticks = apply_logscale(m, ticks, linthresh=1)

    #### Color map #####
    cmap = load_cmap(params["cmap"], params["logscale"])

    if verbose:
        # Terminal ouput
        print(f'FWHM: {fwhm}')
        print(f'nside: {nside}')
        print(f'Ticks: {ticklabels}')
        print(f'Unit: {params["unit"]}')
        print(f'Title: {params["title"]}')
        
    # Plot image
    image = plt.pcolormesh(
        lon[::-1],
        lat,
        grid_map,
        vmin=ticks[0],
        vmax=ticks[-1],
        rasterized=True,
        cmap=cmap,
        shading="auto",
    )

    #### Graticule ####
    if graticule: apply_graticule(ax, plt.gcf().get_size_inches()[0])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    for i in ["title","unit","left_title"]:
        if params[i] and params[i]!="": 
            params[i]=r"$"+params[i]+"$"

    #### Colorbar ####
    if colorbar:
        apply_colorbar(
            fig, ax, image, ticks, ticklabels, params["unit"], 
            fontsize, linthresh=1, logscale=params["logscale"])

    #### Right Title ####
    plt.text(
        4.5,
        1.1,
        params["title"],
        ha="center",
        va="center",
        fontsize=fontsize)

    #### Left Title (stokes parameter label by default) ####
    plt.text(
        -4.5,
        1.1,
        params["left_title"],
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    return fig, ax

def apply_colorbar(fig, ax, image, ticks, ticklabels, unit, fontsize, linthresh, logscale=False):
    """
    This function applies a colorbar to the figure and formats the ticks.
    """
    from matplotlib.ticker import FuncFormatter

    cb = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.4, pad=0.04, ticks=ticks, format=FuncFormatter(fmt),)
    cb.ax.set_xticklabels(ticklabels)
    cb.ax.xaxis.set_label_text(unit)
    cb.ax.xaxis.label.set_size(fontsize)
    if logscale:
        linticks = np.linspace(-1, 1, 3)*linthresh
        logmin = np.round(ticks[0])
        logmax = np.round(ticks[-1])

        logticks_min = -10**np.arange(0, abs(logmin)+1)
        logticks_max = 10**np.arange(0, logmax+1)
        ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))
        #cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks

        logticks = symlog(ticks_, linthresh)
        logticks = [x for x in logticks if x not in ticks]
        cb.set_ticks(np.concatenate((ticks,logticks))) # Set major ticks
        cb.ax.set_xticklabels(ticklabels + ['']*len(logticks))

        minorticks = np.linspace(-linthresh, linthresh, 5)
        minorticks2 = np.arange(2,10)*linthresh

        for i in range(len(logticks_min)):
            minorticks = np.concatenate((-10**i*minorticks2,minorticks))
        for i in range(len(logticks_max)):
            minorticks = np.concatenate((minorticks, 10**i*minorticks2))

        minorticks = symlog(minorticks, linthresh)
        minorticks = minorticks[ (minorticks >= ticks[0]) & ( minorticks<= ticks[-1]) ] 
        cb.ax.xaxis.set_ticks(minorticks, minor=True)

    cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize-2,)
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb

def apply_graticule(ax, width):
    """
    This function applies a graticule to the figure
    """
    #### Formatting ######    
    from matplotlib.projections.geo import GeoAxes
    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        #shifts labelling by pi
        #Shifts labelling from -180,180 to 0-360
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2 * np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    ax.set_longitude_grid(60)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))
    if width < 10:
        ax.set_latitude_grid(45)
        ax.set_longitude_grid_ends(90)            
        ax.grid(True)

def project_map(nside, xsize, ysize,):
    """
    This function generates the projection grid for matplotlib
    """
    theta = np.linspace(np.pi, 0, ysize)
    phi = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))
    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)
    return grid_pix, longitude, latitude
