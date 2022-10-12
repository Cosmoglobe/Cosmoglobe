# This file contains useful functions used across different plotting scripts.
from cmath import inf
from re import A, T
import warnings

from numpy.core.numeric import NaN
from .. import data as data_dir
from cosmoglobe.sky.model import SkyModel
from cosmoglobe.h5.chain import Chain
from cosmoglobe.sky._units import cmb_equivalencies, Unit

import cmasher
from rich import print
import numpy as np
import healpy as hp
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import colorConverter, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon
from matplotlib import _pylab_helpers
from pathlib import Path
import json

DEFAULT_FONTSIZES = {
    "xlabel": 11,
    "ylabel": 11,
    "xtick_label": 8,
    "ytick_label": 8,
    "title": 12,
    "cbar_label": 11,
    "cbar_tick_label": 11,
    "llabel": 11,
    "rlabel": 11,
}

STOKES = [
    "I",
    "Q",
    "U",
]


def set_style(darkmode=False, font="serif"):
    """
    This function sets the color parameter and text style
    """
    if font == "serif":
        plt.rc(
            "font",
            family="serif",
        )
        plt.rcParams["mathtext.fontset"] = "stix"
    plt.rc(
        "text.latex",
        preamble=r"\usepackage{sfmath}",
    )

    tex_fonts = {
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
    plt.rcParams.update(
        tex_fonts,
    )

    ## If darkmode plot (white frames), change colors:
    params = [
        "text.color",
        "axes.edgecolor",
        "axes.labelcolor",
        "xtick.color",
        "ytick.color",
        "grid.color",
        "legend.facecolor",
        "legend.edgecolor",
    ]
    if darkmode:
        for p in params:
            plt.rcParams[p] = "white"
    else:
        for p in params:
            plt.rcParams[p] = "black"


def get_figure_width(width=600, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    # Proper scaling
    fig_dim = (fig_width_in, fig_height_in)
    #print(f"Using dimensions: {fig_dim}")
    return fig_dim


def make_fig(figsize, fignum, hold, sub, reuse_axes, darkmode=False, projection=None, fraction=0.5):
    """
    Create matplotlib figure, add subplot, use current axes etc.
    """

    if figsize is None:
        figsize = get_figure_width(fraction=fraction)

    #  From healpy
    nrows, ncols, idx = (1, 1, 1)
    width, height = figsize
    if not (hold or sub or reuse_axes):
        # Set general style parameters
        set_style(darkmode)

        fig = plt.figure(fignum, figsize=(width, height))
    elif hold:
        fig = plt.gcf()
        # left, bottom, right, top = np.array(fig.gca().get_position()).ravel()
        fig.delaxes(fig.gca())
    elif reuse_axes:
        fig = plt.gcf()
    else:  # using subplot syntax
        if hasattr(sub, "__len__"):
            nrows, ncols, idx = sub
        else:
            nrows, ncols, idx = sub // 100, (sub % 100) // 10, (sub % 10)

        # If figure exists, get. If not, create.
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            fig = figManager.canvas.figure
        else:
            # Set general style parameters
            set_style(darkmode)
            figsize = (width * ncols, height * nrows)
            fig = plt.figure(figsize=figsize)

        if idx < 1 or idx > ncols * nrows:
            raise ValueError("Wrong values for subplot: %d, %d, %d" % (nrows, ncols, idx))

    if reuse_axes:
        ax = fig.gca()
    else:
        ax = fig.add_subplot(int(f"{nrows}{ncols}{idx}"), projection=projection)

    return fig, ax


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
    if abs(x) >= 1e4 or (abs(x) <= 1e-3 and abs(x) > 0):
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        if float(a) == 1.00:
            return r"$10^{" + str(b) + "}$"
        elif float(a) == -1.00:
            return r"$-10^{" + str(b) + "}$"
        else:
            return rf"${a} \cdot 10^{b}$"
    elif abs(x) > 1e1 or (float(abs(x))).is_integer():
        return rf"${int(x):d}$"
    else:
        x = round(x, 2)
        return rf"${x}$"


def format_list(items):
    return [fmt(i, 1) for i in items]


def symlog(m, linthresh=1.0):
    """
    This is the semi-logarithmic function used when logscale=True
    """
    if linthresh == 0.0:
        return np.sign(m) * np.log10(1e-20 + abs(m))

    # Extra fact of 2 ln 10 makes symlog(m) = m in linear regime
    m = m / linthresh / (2 * np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))


def legend_positions(input):
    """
    Calculate position of labels to the right in plot...
    """
    positions = input[-1, :]

    def push(dpush):
        """
        ...by puting them to the last y value and
        pushing until no overlap
        """
        collisions = 0
        for column1, value1 in enumerate(positions):
            for column2, value2 in enumerate(positions):
                if column1 != column2:
                    dist = abs(value1 - value2)
                    if dist < dpush * 10:
                        collisions += 1
                        if value1 < value2:
                            positions[column1] -= dpush
                            positions[column2] += dpush
                        else:
                            positions[column1] += dpush
                            positions[column2] -= dpush
                            return True

    min_ = np.min(np.min(input))
    max_ = np.max(np.max(input))
    dpush = np.abs(max_ - min_) * 0.01

    pushings = 0
    while True:
        if pushings == 1000:
            dpush *= 10
            pushings = 0
        pushed = push(dpush)
        if not pushed:
            break
        if pushings > 100:
            break
        pushings += 1
    return positions


def gradient_fill(x, y, fill_color=None, ax=None, alpha=1.0, invert=False, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """

    if ax is None:
        ax = plt.gca()

    (line,) = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    # alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if invert:
        ymin, ymax = (ymax, ymin)
    im = ax.imshow(z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="lower", zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def gradient_fill_between(ax, x, y1, y2, color="#ffa15a", alpha=0.8):
    N = 100
    y = np.zeros((N, len(y1)))
    for i in range(len(y1)):
        y[:, i] = np.linspace(y1[i], y2[i], N)

    alpha = np.linspace(0, alpha, 100) ** 1.5
    for i in range(1, N):
        ax.fill_between(x, y[i - 1], y[i], color=color, alpha=alpha[i], zorder=-10)
        ax.set_rasterization_zorder(-1)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def apply_logscale(m, ticks, linthresh=1):
    """
    This function converts the data to logscale,
    and also converts the tick locations correspondingly
    NOTE: This should ideally be done for the colorbar, not the data.
    """

    print(f"[magenta]Applying semi-logscale with linear threshold={linthresh}[/magenta]")
    m = symlog(m, linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i, linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks


def standalone_colorbar(
    cmap,
    ticks,
    ticklabels=None,
    unit=None,
    fontsize=None,
    norm=None,
    shrink=0.3,
    darkmode=False,
    width=2,
    extend=None,
):

    fig = plt.figure(figsize=(width, width / 4))
    set_style(darkmode)
    cmap = load_cmap(cmap)
    img = plt.imshow(np.array([[0, 1]]), cmap=cmap)
    plt.gca().set_visible(False)
    ax = plt.axes([0.1, 0.7, 0.8, 0.2])  # LBWH

    if fontsize is None:
        fontsize = DEFAULT_FONTSIZES

    if ticklabels is None:
        ticklabels = format_list(ticks)

    apply_colorbar(
        fig,
        ax,
        None,
        ticks,
        ticklabels,
        unit,
        fontsize=fontsize,
        norm=norm,
        cbar_shrink=shrink,
        cbar_pad=0.0,
        cmap=cmap,
        extend=extend
    )


def apply_colorbar(
    fig,
    ax,
    image,
    ticks,
    ticklabels,
    unit,
    fontsize=None,
    linthresh=1,
    norm=None,
    cbar_pad=0.04,
    cbar_shrink=0.3,
    cmap=None,
    orientation="horizontal",
    extend=None
):
    """
    This function applies a colorbar to the figure and formats the ticks.
    """

    if image is None and cmap is not None:
        norm = mpl.colors.Normalize(vmin=ticks[0], vmax=ticks[-1])
        cb = mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            orientation=orientation,
            ticks=ticks,
            label=unit,
            extend=extend
        )
    else:
        cb = fig.colorbar(
            image,
            # ax=ax,
            orientation=orientation,
            shrink=cbar_shrink,
            pad=cbar_pad,
        )
    if fontsize is None:
        fontsize = DEFAULT_FONTSIZES
    if isinstance(unit, u.UnitBase):
        unit = unit.to_string("latex")
    #if orientation == 'horizontal':
    #   cb.ax.set_xticklabels(ticklabels, size=fontsize["cbar_tick_label"])
    #   cb.ax.xaxis.set_label_text(unit, size=fontsize["cbar_label"])
    #elif orientation == 'vertical':
    #   cb.ax.set_yticklabels(ticklabels, size=fontsize["cbar_tick_label"])
    #   cb.ax.yaxis.set_label_text(unit, size=fontsize["cbar_label"])

    # cb = plt.gca().collections[-1].colorbar
    # labels = cb.ax.xaxis.get_ticklabels()
    ##ticks = cb.get_ticks()
    # N = len(labels)
    # for i, label in enumerate(labels):
    #    if i in [0, N//2, N-1]:
    #        continue
    #    label.set_visible(False)

    if norm in ["linlog", "log"]:
        """
        Make logarithmic tick markers manually
        """
        linticks = np.linspace(-1, 1, 3) * linthresh if linthresh > 0.0 else [0]
        logmin = np.round(ticks[0])
        logmax = np.round(ticks[-1])
        logticks_min = -(10 ** np.arange(0, abs(logmin) + 1))
        logticks_max = 10 ** np.arange(0, logmax + 1)
        ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))

        # cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks
        logticks = symlog(ticks_, linthresh)
        logticks = [x for x in logticks if x not in ticks]
        if orientation == "horizontal":
            cb.set_ticks(np.concatenate((ticks, logticks)))  # Set major ticks
            cb.ax.set_xticklabels(ticklabels + [""] * len(logticks))
        elif orientation == "vertical":
            # cb.ax.yaxis.set_ticks(np.concatenate((ticks, logticks)))  # Set major ticks
            # cb.ax.set_yticklabels(ticklabels + [""] * len(logticks))
            pass

        # Minor tick log markers

        # Linear range for linlog plotting
        minorticks = np.linspace(-linthresh, linthresh, 5) if linthresh > 0.0 else [0]

        # Make first range of logarithmic minor ticks
        lt = 1 if norm == "log" else linthresh
        minorticks2 = np.arange(2, 10) * lt

        # Create range of minor ticks from min value to negative max
        for i in range(len(logticks_min)):
            minorticks = np.concatenate((-(10**i) * minorticks2, minorticks))

        # Create range of minor ticks from min value to positive max
        for i in range(len(logticks_max)):
            minorticks = np.concatenate((minorticks, (10**i) * minorticks2))

        # Convert values to true logarithmic values
        minorticks = symlog(minorticks, linthresh)

        # Remove values outside of range
        minorticks = minorticks[(minorticks >= ticks[0]) & (minorticks <= ticks[-1])]
        if orientation == "horizontal":
            cb.ax.xaxis.set_ticks(minorticks, minor=True)
        elif orientation == "vertical":
            cb.ax.yaxis.set_ticks(minorticks, minor=True)
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    if orientation == "horizontal":
        cb.ax.tick_params(
            which="both",
            axis="x",
            direction="in",
            color="#3d3d3d",
        )
        cb.ax.xaxis.labelpad = 0
    elif orientation == "vertical":
        cb.ax.tick_params(
            which="both",
            axis="y",
            direction="in",
            color="#3d3d3d",
        )
        cb.ax.yaxis.labelpad = 0

        ylabels = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ylabels, Rotation=90)

    return  # cb


def load_cmap(cmap):
    """
    This function takes the colormap label and loads the colormap object
    """

    if cmap is None:
        cmap = "planck"

    if cmap in ["planck", "planck_log", "wmap"]:
        cmap_path =  cmap_path = Path(data_dir.__path__[0]) / f"{cmap}_cmap.dat"
        cmap = ListedColormap(np.loadtxt(cmap_path) / 255.0, cmap)
    elif cmap.startswith("black2"):
        cmap = LinearSegmentedColormap.from_list(cmap, cmap.split("2"))
    else:
        try:
            cmap = eval(f"cmasher.{cmap}")
        except:
            try:

                cmap = eval(f"cm.{cmap}")
            except:

                cmap = plt.get_cmap(cmap)

    return cmap


def get_data(input, sig, comp, freq, fwhm, nside=None, sample=-1):
    # Parsing component string
    fwhm_ = None
    specparam = "amp"
    if comp is not None:
        comp, *specparam_ = comp.split()
        if specparam_:
            specparam = specparam_[0]

    # If map is string, read data
    if isinstance(input, str):
        # If string input, then read either hdf or fits
        if input.endswith(".h5"):
            if comp is None:
                if freq is None:
                    print("[bold magenta]Warning! Neither frequency nor component selected. Plotting sky at 70GHz[/bold magenta]")
                data = SkyModel.from_chain(input, components=comp, nside=nside, samples=sample)
            else:
                if freq is not None:
                    # If frequency is specified, simulate sky with model.
                    data = SkyModel.from_chain(input, components=comp, nside=nside, samples=sample)
                else:
                    chain = Chain(
                        input,
                    )
                    lmax = chain.get(f"{comp}/{specparam}_lmax", samples=sample)
                    alms = chain.get(f"{comp}/{specparam}_alm", samples=sample)
                    fwhm_ = chain.parameters[comp]["fwhm"] * u.arcmin
                    if nside is None:
                        nside = chain.parameters[comp]["nside"]
                    pol = True if specparam == "amp" and alms.shape[0] == 3 else False
                    # PROBLEM! Get issue with nside when udgrading directly to low nside
                    data = hp.alm2map(
                        alms,
                        nside=nside,
                        fwhm=fwhm_.to(u.rad).value,
                        lmax=lmax,
                        mmax=lmax,
                        pol=pol,
                    )

        elif input.endswith(".fits"):
            data = hp.read_map(input, field=sig)
        else:
            raise ValueError("Input file must be .fits or .h5")
    else:
        data = input

    if isinstance(data, SkyModel):
        """
        Get data from model object with frequency scaling
        """
        if comp is None:
            if freq is None:
                print("[bold magenta]Warning! Neither frequency nor component selected. Plotting sky at 70GHz[/bold magenta]")
                freq = 70 * u.GHz
            comp = "freqmap"
            m = data(freq, fwhm=fwhm)
        else:
            if specparam == "amp":
                if freq is None:
                    # Get map at reference frequency if freq not specified
                    freq_ref = data[comp].freq_ref
                    freq = (freq_ref.value).squeeze()
                    m = data[comp].amp
                    if comp == "radio":
                        m = data(freq_ref, fwhm=fwhm, components=[comp])
                    
                    try:
                        freq = np.round(freq[sig], 5) * freq_ref.unit
                    except IndexError:
                        freq = np.round(freq, 5) * freq_ref.unit
                else:
                    # If freq is specified, scale with sky model
                    m = data(freq, fwhm=fwhm, components=[comp])
            else:
                m = data[comp].spectral_parameters[specparam]

                if len(m[sig]) == 1:
                    warnings.warn("Same value across the whole sky, mapping to array of length Npix")
                    m = np.full(hp.nside2npix(data.nside), m[sig])

    elif isinstance(data, np.ndarray):
        m = data
        if m.ndim == 1 and sig > 0 and not isinstance(input, str):
            print(f"[magenta]Input array is 1d, specified signal will only be used for labeling, make sure this is correct.[/magenta]")
    else:
        raise TypeError(f"Type {type(data)} of data not supported" f"Supports numpy array, cosmoglobe model object or fits file string")

    # Make sure it is a 1d array
    if m.ndim > 1:
        if sig>np.shape(m)[0]-1:
            warnings.warn(f"Index {sig} out of data range {np.shape(m)}, returning empty map")
            m = np.zeros(np.shape(m[-1]))+hp.UNSEEN
        else:
            m = m[sig]


    # Convert astropy map to numpy array
    if isinstance(m, u.Quantity):
        m = m.value

    # ud_grade map
    if nside is not None and nside != hp.get_nside(m):
        print("ud_grading map")
        m = hp.ud_grade(m, nside)

    # Smooth map
    if fwhm > 0.0 and fwhm_ is None:
        m = hp.smoothing(m, fwhm.to(u.rad).value)

    return m, comp, freq, hp.get_nside(m)


def get_params(**params):
    """
    This parses the autoparams json file for automatically setting parameters
    for an identified sky component.
    """
    freq = params["freq_ref"]
    # Convert astropy unit to latex
    if isinstance(params["unit"], u.UnitBase):
        params["unit"] = params["unit"].to_string("latex")

    # Autoset plotting parameters if component is specified
    if params["comp"] is not None:
        with open(Path(data_dir.__path__[0]) / "autoparams.json", "r") as f:
            autoparams = json.load(f)
        try:
            autoparams = autoparams[params["comp"]]
        except KeyError:
            print(f"[magenta]Component label {params['comp']} not found. Using unidentified profile.[/magenta]")
            print(f"available keys are: {autoparams.keys()}")
            autoparams = autoparams["unidentified"]

        # Replace default values if None
        params = {key: (autoparams[key] if key in autoparams and value is None else value) for key, value in params.items()}

        # NOTE: This is really ugly code for selecting pol parameters
        # Pick polarization or temperature
        l = 0 if params["sig"] < 1 else -1
        for key in ["cmap", "ticks", "freq_ref"]:
            if isinstance(params[key], list):
                if key == "ticks":
                    if isinstance(params[key][0], list):
                        params[key] = params[key][l]
                else:
                    params[key] = params[key][l]

        if params["freq_ref"] is not None:
            params["freq_ref"] * u.GHz

        if params["llabel"] is None:
            params["llabel"] = ["I", "Q", "U"][params["sig"]]
        specials = ["residual", "freqmap", "bpcorr", "smap"]
        if any(j in params["comp"] for j in specials):
            if freq is not None:
                params["rlabel"] = params["rlabel"] + "{" + f'{("%.5f" % freq.value).rstrip("0").rstrip(".")}' + "}"
            else:
                params["rlabel"] = None
                warnings.warn(f'Specify frequency with -freq for automatic frequency labeling with the "freqmap" profile')

        """
        # NOTE: This probably doesnt work with the current state of the comp
        # keyword. Rewrite somehow.
        if "rms" in params["comp"]:
            params["rlabel"] += "^{\mathrm{RMS}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "stddev" in params["comp"]:
            params["rlabel"] = "\sigma_{\mathrm{" + params["rlabel"] + "}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "mean" in params["comp"]:
            params["rlabel"] = "\langle " + params["rlabel"] + "\rangle"
        """

        # Specify at which frequency we are observing in unit lable
        if freq is not None and params["unit"] is not None:
            params["unit"] = f'{params["unit"]}\,@\,{("%.5f" % freq.value).rstrip("0").rstrip(".")}' + "\,\mathrm{GHz}"

        # In the case that a difference frequency than the reference
        # is requested for a component map.
        if params["ticks"] == None:
            if freq is not None and params["freq_ref"] != freq.value and params["comp"] not in specials:
                warnings.warn(f"Input frequency is different from reference, autosetting ticks")
                print(f'input: {freq}, reference: {params["freq_ref"]}')
                params["ticks"] = "auto"

    # If ticks is not set automatically, try inputs
    if params["ticks"] is None:
        params["ticks"] = [params["min"], params["max"]]

    # if ticks is not specified to be auto in function call
    if params["ticks"] != "auto":
        # If range is not specified in function call
        if params["rng"] is not None:
            params["ticks"] = [-params["rng"], 0.0, params["rng"]]
        else:
            # If min is not specified in function call
            if params["min"] is not None:
                # If ticks is set to be automatic from autoparams
                if params["ticks"] == "auto":
                    params["ticks"] = [None, None]  # (These will be set later)
                params["ticks"][0] = params["min"]
            # If max is not specified in function call
            if params["max"] is not None:
                # If tick is set to be automatic from atoparams
                if params["ticks"] == "auto":
                    params["ticks"] = [None, None]  # (These will be set later)
                params["ticks"][-1] = params["max"]

    # Ticks and ticklabels
    if params["ticks"] == "auto" or "auto" in params["ticks"]:
        params["ticks"] = get_percentile(params["data"], 99)
    elif None in params["ticks"]:
        pmin, pmax = get_percentile(params["data"], 99)
        if params["ticks"][0] is None:
            params["ticks"][0] = pmin
        if params["ticks"][-1] is None:
            params["ticks"][-1] = pmax

    # Math text in labels
    for key in ["rlabel", "llabel", "unit"]:
        if params[key] and params[key] != "":
            if "$" not in params[key]:
                params[key] = r"$" + params[key] + "$"

    # Special case if dipole is detected in freqmap
    if params["comp"] == "freqmap" and params["min"] is None and params["max"] is None and params["rng"] is None:
        # if colorbar is super-saturated
        while len(params["data"][abs(params["data"]) > params["ticks"][-1]]) / len(params["data"].ravel()) > 0.7:
            params["ticks"] = [tick * 10 for tick in params["ticks"]]
            print("[magenta]Colormap saturated. Expanding color-range.[/magenta]")

    # Create ticklabels from final ticks
    params["ticklabels"] = format_list(params["ticks"])

    # Set planck as default cmap
    if params["cmap"] is None:
        params["cmap"] = "planck"
    
    return params


def mask_map(m, mask):
    """
    Apply a mask to a map
    """

    m = hp.ma(m)
    m.mask = mask
    # Alternative

    # mask = mask*np.NaN
    # m *= mask
    return m


def create_70GHz_mask(sky_frac, nside=256, pol=False):
    """
    Creates a mask from a 70GHz frequency map at nside=256 for a
    threshold corresponding to a given sky fraction (in %). The 70GGz
    map is chosen due to it containing a large portion of all low-frequency
    sky components.
    Parameters
    ----------
    sky_frac : float
        Sky fraction in percentage.
    Returns
    -------
    mask : numpy.ndarray
        Mask covering the sky for a given sky fraction.
    """
    # Read amplitude map to use for thresholding
    field = (1, 2) if pol else 0
    template = hp.read_map(
        Path(data_dir.__path__[0]) / "mask_template_n256.fits",
        dtype=np.float64,
        field=field,
    )
    if pol:
        template = np.sqrt(template[0] ** 2 + template[1] ** 2)
    template = hp.ma(template)

    if nside != 256:
        template = hp.ud_grade(template, nside)

    # Masking based on sky fraction is not trivial. Here we manually compute
    # the sky fraction by masking all pixels with amplitudes larger than a
    # given percentage of the maximum map amplitude. The pixels masks then
    # correspond to a sky fraction. We tabulate the amplitude percentage and
    # sky fraction for a range, and interpolate from this table.
    amp_percentages = np.flip(np.arange(1, 101))
    fracs = []
    mask = np.zeros(len(template), dtype=np.bool)

    for i in range(len(amp_percentages)):
        mask = np.zeros(len(template), dtype=np.bool)
        masked_template = np.abs(hp.ma(template))
        mask[np.where(np.log(masked_template) > (amp_percentages[i] / 100) * np.nanmax(np.log(masked_template)))] = 1
        masked_template.mask = mask

        frac = ((len(masked_template) - masked_template.count()) / len(masked_template)) * 100
        fracs.append(frac)

    amp_percentage = np.interp(100 - sky_frac, fracs, amp_percentages)

    mask = np.zeros(len(template), dtype=np.bool)
    masked_template = np.abs(hp.ma(template))
    mask[np.where(np.log(masked_template) > (amp_percentage / 100) * np.nanmax(np.log(masked_template)))] = 1

    return mask


def seds_from_model(nu, model, nside=None, pol=False, sky_fractions=(25, 85)):
    ignore_comps = ["radio"]
    comps = model.components
    if nside is None or nside == model.nside:
        nside = model.nside
        udgrade = False
    else:
        udgrade = True

    masks = np.zeros((len(sky_fractions), hp.nside2npix(nside)))

    for i, sky_frac in enumerate(sky_fractions):
        masks[i] = create_70GHz_mask(sky_frac, nside, pol=pol)

    # SED dictionary with T and P, skyfractions and the value per nu
    seds = {comp: np.zeros((2, len(sky_fractions), len(nu))) for comp in model.components}
    for key, value in comps.items():
        if key in ignore_comps:
            continue

        for i, mask in enumerate(masks):
            specinds = {}
            for specind, m in value.spectral_parameters.items():
                if m.shape[-1] != 1:
                    m = np.mean(mask_map(m, mask), axis=1).reshape(-1, 1) * value.spectral_parameters[specind].unit
                specinds[specind] = m

            amp = hp.ud_grade(value.amp, nside) if udgrade else value.amp
            amp[0, amp[0] < 0.0] = 0.0  # abs(amp)
            amp = mask_map(amp, mask)

            amp_pol = 0
            if pol:
                if amp.shape[0] > 1:
                    amp_pol = rms_amp(np.sqrt(amp[1] ** 2 + amp[2] ** 2))
            amp = rms_amp(amp[0])

            if key == "cmb":
                freq_scaling = (np.ones(len(nu)) * Unit("uK_CMB")).to("uK_RJ", equivalencies=cmb_equivalencies(nu * u.GHz))
                seds[key][0, i, :] = 45 * freq_scaling  # TEMP
                seds[key][1, i, :] = 0.67 * freq_scaling  # POL
            else:
                freq_scaling = value.get_freq_scaling(nu * u.GHz, **specinds)
                k = 1 if freq_scaling.shape[0] > 1 else 0
                seds[key][0, i, :] = amp * freq_scaling[0]
                seds[key][1, i, :] = amp_pol * freq_scaling[k]

    return seds


def rms_amp(m):
    n = len(m) - np.sum(m.mask)
    m -= np.mean(m)
    return np.sqrt(np.sum(m**2) / n)
