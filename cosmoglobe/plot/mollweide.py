import click
import sys
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

hp.disable_warnings()
# Fix for macos openMP duplicate bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def mollplot(
    map_,
    auto=False,
    min=False,
    max=False,
    mid=[],
    rng="auto",
    colorbar=False,
    graticule=False,
    lmax=None,
    fwhm=0.0,
    mask=None,
    mfill=None,
    sig=[0,],
    logscale=False,
    remove_dipole=None,
    remove_monopole=None,
    cmap=None,
    title=None,
    ltitle=None,
    outdir=".",
    outname=None,
    size="m",
    white_background=False,
    darkmode=False,
    png=False,
    cfontsize=11,
    dpi=300,
    xsize=2000,
    hires=False,
    gif=False,
):
    """
    Plots a fits file, a h5 data structure or a data array in Mollview with
    pretty formatting. Option of autodecting component base on file-string with
    json look-up table.

    Parameters
    ----------
    map_ : cosmoglobe map object
        cosmoglobe map object with I (optional Q and U)
    auto : bool
        Toggle parameter autodetector. Automatic identification of plotting
        parameters based on information from autoparams.json default = False
    min :  str
        Minimum value of colorbar. Overrides autodetector.
        default = False
    max :  str
        Maximum value of colorbar. Overrides autodetector.
        default = False
    mid : list
        Adds tick values between min and max. Accepts multiple ex. -mid 2 -mid
        3. default = []
    range : str
        Min and max value around 0 (ex. 10).
        Also compatible with "auto": sets to 97.5 percentile of data.
        Also compatible with "minmax": sets to min and max values.
        default = "auto"
    colorbar : bool
        Adds a colorbar, and "cb" to output filename.
        default = False
    graticule : bool
        Adds graticule to figure.
        default = False
    lmax : float
        Set the lmax of alm input data from hdf.
        If reading hdf and not set, this is automatically found.
        default = None
    fwhm : float
        Optional map smoothing. FWHM of gaussian smoothing in arcmin.
        default = 0.0
    mask : str
        Apply a mask file (supply filename) to data (uses signal 0).
        default = None
    mfill : str
        Color to fill masked area, for example "gray".
        Transparent by default.
        defualt : None
    sig : list of int
        Signal indices to be plotted. 0, 1, 2 interprated as IQU.
        Supports multiple, such that -sig 1 -sig 2.
        default = [0,]
    remove_dipole : str
        Fits and removes a dipole. Specify mask for fit, or "auto".
        default = None
    remove_monopole : str
        Fits and removes a monopole. Specify mask for fit, or "auto".
        default = None
    logscale : str
        Normalizes data using a semi-logscale linear between -1 and 1.
        Autodetector uses this sometimes, you will be warned.
        default = None
    size : str
        Size of output maps. 1/3, 1/2 and full page width (8.8/12/18cm) [ex. x,
        s, m or l, or ex. slm for all]" default = "m"
    white_background : bool
        Sets white background. Transparent by default.
        default = False
    darkmode : bool
        Plots all outlines in white for dark backgrounds, and adds "dark" in
        filename.
        default = False
    png : bool
        Saves file as png as opposed to PDF. Overwritten by outname extension.
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
    unit : str
        Sets uniit under color bar. Has LaTeX functionaliity (ex. $\mu K$".
        default = None
    scale : float
        Scale data by multiplicatiive factor. [ex. 1e-6 for muK to K
        default = None
    outdir : path
        Optiional different output directory.
        default = "."
    outname : str
        Output filename. If not specified, descriptive filename will be
        generated from chosen params. Extension determines filetype and
        overrides -png flag.
        default = None
    gif : bool
        Make gif is input is list of data.
        defaullt = False
    oldfont : bool
        Use DejaVu font as oppoosed to Times.
        default = False
    fontsize : int
        Fontsize
        default = 11
    dpi : int
        DPI (Dots per inch). For figure resolution.
        default = 300
    xsize : int
        Figure width in pixels. Uses xsize*(xsize/2)
        default = 2000
    hires : bool
        For high resolution figures. Sets dpi to 3000 and xsize to 10000.
        default = False
    """
    # Check if input is cosmoglobe map object
    if not isinstance(map_, (IQU_map)):
        map_ = map_.to_IQU
    
    print(f"Plotting {map_.label}")

    # Set plotting rcParams
    xsize, dpi = set_style(hires, outname, darkmode)

    






















def set_style(hires, outname, darkmode):
    # rcParameters
    plt.rcParams["mathtext.fontset"] =  
    plt.rcParams["axes.linewidth"] = 1

    # Preset high resolution parameters for superbig plots
    if hires:
        xsize=10000
        dpi=1000

    ## Set backend based on output filename or     
    plt.rcParams["backend"] = "agg" if png else "pdf"
    if outname:
        plt.rcParams["backend"] = "agg" if outname.endswith("png") else "pdf"

    ## If darkmode plot (white frames), change colors:
    if darkmode:
        params = ["text.color", "axes.facecolor", "axes.edgecolor", "axes.labelcolor", "xtick.color",
                  "ytick.color", "grid.color", "legend.facecolor", "legend.edgecolor"]
        for p in params:
            plt.rcParams[p] = "white"

    return xsize, dpi
