# This file contains useful functions used across different plotting scripts.
from re import A, T
import warnings
from .. import data as data_dir
from cosmoglobe.sky.model import Model
from cosmoglobe.h5.model import model_from_chain
from cosmoglobe.h5.chain import Chain

import cmasher
from rich import print
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
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
    "cbar_tick_label": 9,
    "left_label": 11,
    "right_label": 11,
}
FIGURE_WIDTHS = {
    "x": 2.75,
    "s": 3.5,
    "m": 4.7,
    "l": 7,
}
STOKES = [
    "I",
    "Q",
    "U",
]


def set_style(darkmode=False):
    """
    This function sets the color parameter and text style
    """
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
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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


def make_fig(
    figsize, fignum, hold, subplot, reuse_axes, darkmode=False, projection=None
):
    """
    Create matplotlib figure, add subplot, use current axes etc.
    """

    if figsize is None:
        fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        figsize = [fig_width, fig_height]

    #  From healpy
    nrows, ncols, idx = (1, 1, 1)
    width, height = figsize
    if not (hold or subplot or reuse_axes):
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
        if hasattr(subplot, "__len__"):
            nrows, ncols, idx = subplot
        else:
            nrows, ncols, idx = subplot // 100, (subplot % 100) // 10, (subplot % 10)

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
            raise ValueError(
                "Wrong values for subplot: %d, %d, %d" % (nrows, ncols, idx)
            )

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


def format_list(items):
    return [fmt(i, 1) for i in items]


def symlog(m, linthresh=1.0):
    """
    This is the semi-logarithmic function used when logscale=True
    """
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
    im = ax.imshow(
        z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="lower", zorder=zorder
    )

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def gradient_fill_between(ax, x, y1, y2, color="#ffa15a", alpha=0.5):
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


def standalone_colorbar(
    cmap,
    ticks,
    ticklabels=None,
    unit=None,
    fontsize=None,
    norm=None,
    shrink=0.3,
    darkmode=False,
    width=10,
):

    set_style(darkmode)
    if fontsize is None:
        fontsize = DEFAULT_FONTSIZES
    plt.figure(figsize=(width, 2))
    img = plt.imshow(np.array([[ticks[0], ticks[-1]]]), load_cmap(cmap))
    plt.gca().set_visible(False)
    if ticklabels is None:
        ticklabels = format_list(ticks)
    apply_colorbar(
        plt.gcf(),
        plt.gca(),
        img,
        ticks,
        ticklabels,
        unit,
        fontsize=fontsize,
        norm=norm,
        cbar_shrink=shrink,
        cbar_pad=0.0
    )


def apply_logscale(m, ticks, linthresh=1):
    """
    This function converts the data to logscale,
    and also converts the tick locations correspondingly
    NOTE: This should ideally be done for the colorbar, not the data.
    """

    print("[magenta]Applying semi-logscale[/magenta]")
    m = symlog(m, linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i, linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks


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
):
    """
    This function applies a colorbar to the figure and formats the ticks.
    """
    from matplotlib.ticker import FuncFormatter

    if fontsize is None:
        fontsize = DEFAULT_FONTSIZES
    cb = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        shrink=cbar_shrink,
        pad=cbar_pad,
        ticks=ticks,
        format=FuncFormatter(fmt),
    )
    cb.ax.set_xticklabels(ticklabels, size=fontsize["cbar_tick_label"])
    if isinstance(unit, u.UnitBase):
        unit = unit.to_string("latex")
    cb.ax.xaxis.set_label_text(unit, size=fontsize["cbar_label"])
    if norm == "log":
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
    )
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb


def load_cmap(cmap):
    """
    This function takes the colormap label and loads the colormap object
    """

    if cmap is None:
        cmap = "planck"
    if "planck" in cmap:
        if "planck_log" in cmap:  # logscale:
            # setup nonlinear colormap

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
                    yi = (yi + 3) / 10.0
                    return self.cmap(yi, alpha)

                def modsinh(self, x):
                    return np.log10(0.5 * (x + np.sqrt(x ** 2 + 4)))

            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap_logscale.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0
            cmap = ListedColormap(planck_cmap, "planck")
            cmap = GlogColormap(cmap)
        else:
            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0
            if cmap.endswith("_r"):
                planck_cmap = planck_cmap[::-1]
            cmap = ListedColormap(planck_cmap, "planck")
    elif "wmap" in cmap:
        cmap_path = Path(data_dir.__path__[0]) / "wmap_cmap.dat"
        wmap_cmap = np.loadtxt(cmap_path) / 255.0
        if cmap.endswith("_r"):
            wmap_cmap = wmap_cmap[::-1]
        cmap = ListedColormap(wmap_cmap, "wmap")
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

    # print("Colormap:" + f" {cmap.name}")
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
                    print(
                        "[bold magenta]Warning! Neither frequency nor component selected. Plotting sky at 70GHz[/bold magenta]"
                    )
                data = model_from_chain(
                    input, components=comp, nside=nside, samples=sample
                )
            else:
                if freq is not None:
                    # If frequency is specified, simulate sky with model.
                    data = model_from_chain(
                        input, components=comp, nside=nside, samples=sample
                    )
                else:
                    chain = Chain(
                        input,
                    )
                    lmax = chain.get(f"{comp}/{specparam}_lmax", samples=sample)
                    alms = chain.get(f"{comp}/{specparam}_alm", samples=sample)
                    fwhm_ = chain.parameters[comp]["fwhm"] * u.arcmin
                    if nside is None: nside = chain.parameters[comp]["nside"]
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

    if isinstance(data, Model):
        """
        Get data from model object with frequency scaling
        """
        if comp is None:
            if freq is None:
                print(
                    "[bold magenta]Warning! Neither frequency nor component selected. Plotting sky at 70GHz[/bold magenta]"
                )
                freq = 70 * u.GHz
            comp = "freqmap"
            m = data(freq, fwhm=fwhm)
        else:
            if specparam == "amp":
                if freq is None:
                    # Get map at reference frequency if freq not specified
                    freq_ref = getattr(data, comp).freq_ref
                    freq = freq_ref.value
                    m = getattr(data, comp).amp
                    if comp == "radio":
                        m = getattr(data, comp)(freq_ref, fwhm=fwhm)
                    try:
                        freq = round(freq.squeeze()[sig], 5) * freq_ref.unit
                    except IndexError:
                        freq = round(freq, 5) * freq_ref.unit
                else:
                    # If freq is specified, scale with sky model
                    m = getattr(data, comp)(freq, fwhm=fwhm)
            else:
                m = getattr(data, comp).spectral_parameters[specparam]

                if len(m[sig]) == 1:
                    warnings.warn(
                        "Same value across the whole sky, mapping to array of length Npix"
                    )
                    m = np.full(hp.nside2npix(data.nside), m[sig])

    elif isinstance(data, np.ndarray):
        m = data
        if m.ndim == 1 and sig > 0 and not isinstance(input, str):
            print(
                f"[magenta]Input array is 1d, specified signal will only be used for labeling, make sure this is correct.[/magenta]"
            )
    else:
        raise TypeError(
            f"Type {type(data)} of data not supported"
            f"Supports numpy array, cosmoglobe model object or fits file string"
        )

    # Make sure it is a 1d array
    if m.ndim > 1:
        m = m[sig]

    # Convert astropy map to numpy array
    if isinstance(m, u.Quantity):
        m = m.value

    # ud_grade map
    if nside is not None and nside != hp.get_nside(m):
        print("Ud_grading map")
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
            print(
                f"[magenta]Component label {params['comp']} not found. Using unidentified profile.[/magenta]"
            )
            print(f"available keys are: {autoparams.keys()}")
            autoparams = autoparams["unidentified"]

        # Replace default values if None
        params = {
            key: (autoparams[key] if key in autoparams and value is None else value)
            for key, value in params.items()
        }

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

        if params["left_label"] is None:
            params["left_label"] = ["I", "Q", "U"][params["sig"]]
        specials = ["residual", "freqmap", "bpcorr", "smap"]
        if any(j in params["comp"] for j in specials):
            if freq is not None:
                params["right_label"] = (
                    params["right_label"]
                    + "{"
                    + f'{("%.5f" % freq.value).rstrip("0").rstrip(".")}'
                    + "}"
                )
            else:
                params["right_label"] = None
                warnings.warn(
                    f'Specify frequency with -freq for automatic frequency labeling with the "freqmap" profile'
                )

        """
        # NOTE: This probably doesnt work with the current state of the comp
        # keyword. Rewrite somehow.
        if "rms" in params["comp"]:
            params["right_label"] += "^{\mathrm{RMS}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "stddev" in params["comp"]:
            params["right_label"] = "\sigma_{\mathrm{" + params["right_label"] + "}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "mean" in params["comp"]:
            params["right_label"] = "\langle " + params["right_label"] + "\rangle"
        """

        # Specify at which frequency we are observing in unit lable
        if freq is not None and params["unit"] is not None:
            params["unit"] = (
                f'{params["unit"]}\,@\,{("%.5f" % freq.value).rstrip("0").rstrip(".")}'
                + "\,\mathrm{GHz}"
            )

        # In the case that a difference frequency than the reference
        # is requested for a component map.
        if params["ticks"] == None:
            if (
                freq is not None
                and params["freq_ref"] != freq.value
                and params["comp"] not in specials
            ):
                warnings.warn(
                    f"Input frequency is different from reference, autosetting ticks"
                )
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
    if params["ticks"] == "auto" or params["norm"] == "hist":
        params["ticks"] = get_percentile(params["data"], 97.5)
    elif None in params["ticks"]:
        pmin, pmax = get_percentile(params["data"], 97.5)
        if params["ticks"][0] is None:
            params["ticks"][0] = pmin
        if params["ticks"][-1] is None:
            params["ticks"][-1] = pmax

    # Math text in labels
    for key in ["right_label", "left_label", "unit"]:
        if params[key] and params[key] != "":
            if "$" not in params[key]:
                params[key] = r"$" + params[key] + "$"

    # Special case if dipole is detected in freqmap
    if (
        params["comp"] == "freqmap"
        and params["min"] is None
        and params["max"] is None
        and params["rng"] is None
    ):
        # if colorbar is super-saturated
        while (
            len(params["data"][abs(params["data"]) > params["ticks"][-1]])
            / len(params["data"].ravel())
            > 0.7
        ):
            params["ticks"] = [tick * 10 for tick in params["ticks"]]
            print("[magenta]Colormap saturated. Expanding color-range.[/magenta]")

    # Create ticklabels from final ticks
    params["ticklabels"] = format_list(params["ticks"])

    return params
