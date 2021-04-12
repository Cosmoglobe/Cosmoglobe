import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import astropy.units as u
from cosmoglobe.tools.map import to_stokes, StokesMap
from pathlib import Path
from .. import data as data_dir

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

    # From healpy
    nrows, ncols, idx = (1,1,1)
    if not (hold or subplot or reuse_axes):
        fig = plt.figure(fignum, figsize=(width,height))
    elif hold:
        fig = plt.gcf()
        left, bottom, right, top = np.array(fig.gca().get_position()).ravel()
        fig.delaxes(fig.gca())
    elif reuse_axes:
        fig = plt.gcf()
    else:  # using subplot syntax
        if hasattr(subplot, "__len__"):
            nrows, ncols, idx = subplot
        else:
            nrows, ncols, idx = subplot // 100, (subplot % 100) // 10, (subplot % 10)
        
        # If figure exists, get. If not, create.
        from matplotlib import _pylab_helpers
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            fig = figManager.canvas.figure
        else:
            figsize = (width*ncols, height*nrows)
            fig = plt.figure(figsize=figsize)

        if idx < 1 or idx > ncols * nrows:
            raise ValueError("Wrong values for subplot: %d, %d, %d" % (nrows, ncols, idx))

    ax = fig.add_subplot(int(f"{nrows}{ncols}{idx}"), projection='mollweide')

    # Translate sig to correct format
    stokes=["I", "Q", "U"]
    if isinstance(sig, str): sig=stokes.index(sig)

    if auto is not None: 
        import json
        with open(Path(data_dir.__path__[0]) /'autoparams.json', 'r') as f:
            autoparams = json.load(f)
        try:
            params = autoparams[auto]
        except:
            print(f"Component label {auto} not found. Using unidentified profile.")
            print(autoparams.keys())
            params = autoparams["unidentified"]

        #If none will be set to IQU
        params["left_title"] = stokes[sig]

        # If l=0 use intensity cmap and ticks, else use QU
        l = 0 if sig<1 else 1
        params["cmap"] = params["cmap"][l]
        params["ticks"] = params["ticks"][l]

        if any(j in auto for j in ["residual", "freqmap", "bpcorr", "smap"]):
            # Add number, such as frequency, to title.
            number = ''.join(filter(lambda i: i.isdigit(), auto))
            params["title"]=params["title"]+"{"+number+"}"

        if "rms" in auto:
            params["title"] += "^{\mathrm{RMS}}"
            params["cmap"] = "neutral"
            params["ticks"] = "auto"
        if "stddev" in auto:
            params["title"] = "\sigma_{\mathrm{" + param["title"] + "}}"
            params["cmap"] = "neutral"
            params["ticks"] = "auto"
        if "mean" in auto:
            params["title"] =  "\langle " + comp["title"] +"\rangle"

        # This is messy
        if title is not None: params["title"] = title
        if ltitle is not None: params["left_title"] = ltitle
        if unit is not None: params["unit"] = unit
        if ticks is not None: params["ticks"] = ticks
        if logscale is not None: params["logscale"] = logscale
        if cmap is not None: params["cmap"] = cmap
    else:
        params = {"title": title, 
                    "left_title": stokes[sig],
                    "unit": unit,
                    "ticks": ticks,
                    "logscale": logscale,
                    "cmap": cmap,
                }

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
    elif remove_mono:
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
    cmap = load_cmap(params["cmap"], logscale)

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

def apply_logscale(m, ticks, linthresh=1):
    """
    This function converts the data to logscale, 
    and also converts the tick locations correspondingly
    """
    print("Applying semi-logscale")
    m = symlog(m,linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i,linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks

def set_style(darkmode):
    """
    This function sets the color parameter and text style
    """
    # rcParameters
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1

    ## If darkmode plot (white frames), change colors:
    if darkmode:
        params = [
            "text.color",
            "axes.facecolor",
            "axes.edgecolor",
            "axes.labelcolor",
            "xtick.color",
            "ytick.color",
            "grid.color",
            "legend.facecolor",
            "legend.edgecolor",
        ]
        for p in params:
            plt.rcParams[p] = "white"

def load_cmap(cmap,logscale):
    """
    This function takes the colormap label and loads the colormap object
    """

    if cmap == None: cmap = "planck"
    if "planck" in cmap:
        if "planck_log" in cmap: #logscale:
            # setup nonlinear colormap
            from matplotlib.colors import LinearSegmentedColormap
            class GlogColormap(LinearSegmentedColormap):
                name = cmap
                def __init__(self, cmap, vmin=-1e3, vmax=1e7):
                    self.cmap = cmap
                    self.N = self.cmap.N
                    self.vmin = vmin
                    self.vmax = vmax

                def is_gray(self):
                    return False

                def __call__(self, xi, alpha=1.0, **kw):
                    x = xi * (self.vmax - self.vmin) + self.vmin
                    yi = self.modsinh(x)
                    # range 0-1
                    yi = (yi + 3)/10.
                    return self.cmap(yi, alpha)

                def modsinh(self, x):
                    return np.log10(0.5*(x + np.sqrt(x**2 + 4)))
            
            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap_logscale.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            cmap = col.ListedColormap(planck_cmap, "planck")
            cmap = GlogColormap(cmap)        
        else:
            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            if cmap.endswith("_r"):
                planck_cmap = planck_cmap[::-1]
            cmap = col.ListedColormap(planck_cmap, "planck")
    elif "wmap" in cmap:
        cmap_path = Path(data_dir.__path__[0]) / "wmap_cmap.dat"
        wmap_cmap = np.loadtxt(cmap_path) / 255.0            
        if cmap.endswith("_r"):
            planck_cmap = planck_cmap[::-1]
        cmap = col.ListedColormap(wmap_cmap, "wmap")
    elif cmap.startswith("q-"):
        import plotly.colors as pcol
        _, clab, *numvals = cmap.split("-")
        colors = getattr(pcol.qualitative, clab)
        if clab=="Plotly":
            #colors.insert(3,colors.pop(-1))
            colors.insert(0,colors.pop(-1))
            colors.insert(3,colors.pop(2))
        try:
            cmap = col.ListedColormap(colors[:int(numvals[0])], f'{clab}-{numvals[0]}')
            print("Using qualitative colormap:" + f" {clab} up to {numvals[0]}")
        except:
            cmap = col.ListedColormap(colors,clab)
            print("Using qualitative colormap:" f" {clab}")
    elif cmap.startswith("black2"):
        cmap = col.LinearSegmentedColormap.from_list(cmap,cmap.split("2"))
    else:
        try:
            import cmasher
            cmap = eval(f"cmasher.{cmap}")
        except:
            try:
                from cmcrameri import cm
                cmap = eval(f"cm.{cmap}")
            except:
                cmap = plt.get_cmap(cmap)

    print("Colormap:" + f" {cmap.name}")
    return cmap

def get_percentile(m, percentile):
    """
    This function gets appropriate min and max 
    values of the data for a given percentile
    """
    vmin = np.percentile(m, 100.0 - percentile)
    vmax = np.percentile(m, percentile)

    vmin = 0.0 if abs(vmin) < 1e-5 else vmin
    vmax = 0.0 if abs(vmax) < 1e-5 else vmax
    return [vmin, vmax]

def fmt(x, pos):
    """
    Format color bar labels
    """
    if abs(x) > 1e4 or (abs(x) <= 1e-3 and abs(x) > 0):
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        if float(a) == 1.00:
            return r"$10^{" + str(b) + "}$"
        elif float(a) == -1.00:
            return r"$-10^{" + str(b) + "}$"
        else:
            return fr"${a} \cdot 10^{b}$"
    elif abs(x) > 1e1 or (float(abs(x))).is_integer():
        return fr"${int(x):d}$"
    else:
        x = round(x, 2)
        return fr"${x}$"

def symlog(m, linthresh=1.0):
    """
    This is the semi-logarithmic function used when logscale=True
    """
    # Extra fact of 2 ln 10 makes symlog(m) = m in linear regime
    m = m/linthresh/(2*np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))

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
