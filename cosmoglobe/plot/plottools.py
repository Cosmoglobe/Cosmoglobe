# This file contains useful functions used across different plotting scripts.
from re import T
import warnings
from .. import data as data_dir

import cmasher
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import colorConverter, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon
from matplotlib import _pylab_helpers
from pathlib import Path
import json


def apply_logscale(m, ticks, linthresh=1):
    """
    This function converts the data to logscale,
    and also converts the tick locations correspondingly
    """
    print("Applying semi-logscale")
    m = symlog(m, linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i, linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks


def set_style(darkmode=False,):
    """
    This function sets the color parameter and text style
    """
    plt.rc(
        "font", family="serif",
    )
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rc(
        "text.latex", preamble=r"\usepackage{sfmath}",
    )

    tex_fonts = {
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    plt.rcParams.update(tex_fonts,)

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


def make_fig(
    figsize, fignum, hold, subplot, reuse_axes, darkmode=False, projection=None,
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


def load_cmap(cmap,):
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
    m = m / linthresh / (2 * np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))


def autoparams(
    comp, sig, right_label, left_label, unit, ticks, min, max, norm, cmap, freq
):
    """
    This parses the autoparams json file for automatically setting parameters
    for an identified sky component.
    """
    if isinstance(unit, u.UnitBase):
        unit = unit.to_string()
    if comp is None:
        params = {
            "right_label": right_label,
            "left_label": left_label,
            "unit": unit,
            "ticks": ticks,
            "norm": norm,
            "cmap": cmap,
            "freq_ref": freq,
        }
    else:
        with open(Path(data_dir.__path__[0]) / "autoparams.json", "r") as f:
            autoparams = json.load(f)
        try:
            params = autoparams[comp]
        except:
            print(f"Component label {comp} not found. Using unidentified profile.")
            print(f"available keys are: {autoparams.keys()}")
            params = autoparams["unidentified"]

        # If l=0 use intensity cmap and ticks, else use QU
        l = 0 if sig < 1 else -1
        params["cmap"] = params["cmap"][l]
        params["ticks"] = params["ticks"][l]
        params["freq_ref"] = params["freq_ref"][l]
        params["left_label"] = ["I", "Q", "U"][sig]

        if params["freq_ref"] is not None:
            params["freq_ref"] * u.GHz

        specials = ["residual", "freqmap", "bpcorr", "smap"]
        if any(j in comp for j in specials):
            params["right_label"] = (
                params["right_label"]
                + "{"
                + f'{("%.5f" % freq.value).rstrip("0").rstrip(".")}'
                + "}"
            )

        if "rms" in comp:
            params["right_label"] += "^{\mathrm{RMS}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "stddev" in comp:
            params["right_label"] = "\sigma_{\mathrm{" + params["right_label"] + "}}"
            params["cmap"] = "neutral"
            params["ticks"] = "comp"
        if "mean" in comp:
            params["right_label"] = "\langle " + comp["right_label"] + "\rangle"

        # Assign values if specified in function call
        if right_label is not None:
            params["right_label"] = right_label
        if left_label is not None:
            params["left_label"] = left_label
        if unit is not None:
            params["unit"] = unit
        if ticks is not None:
            params["ticks"] = ticks
        if norm is not None:
            params["norm"] = norm
        if cmap is not None:
            params["cmap"] = cmap
        if freq is not None and params["unit"] is not None:
            params["unit"] = (
                f'{params["unit"]}\,@\,{("%.5f" % freq.value).rstrip("0").rstrip(".")}'
                + "\,\mathrm{GHz}"
            )
        if ticks == None:
            if (
                freq != None
                and params["freq_ref"] != freq.value
                and comp not in specials
            ):
                warnings.warn(
                    f"Input frequency is different from reference, autosetting ticks"
                )
                print(f'input: {freq}, reference: {params["freq_ref"]}')
                params["ticks"] = "auto"

    if params["ticks"] is None:
        params["ticks"] = [min, max]
    if ticks != "auto":
        if min is not None:
            if params["ticks"] == "auto":
                params["ticks"] = [None, None]
            params["ticks"][0] = min
        if max is not None:
            if params["ticks"] == "auto":
                params["ticks"] = [None, None]
            params["ticks"][-1] = max

    # Math text in labels
    for i in [
        "right_label",
        "left_label",
        "unit",
    ]:
        if params[i] and params[i] != "":
            params[i] = r"$" + params[i] + "$"

    return params


def legend_positions(input,):
    """
    Calculate position of labels to the right in plot...
    """
    positions = input[:, -1]

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
    print(min_, max_, dpush)

    pushings = 0
    while True:
        if pushings == 1000:
            dpush *= 10
            pushings = 0
        pushed = push(dpush)
        if not pushed:
            break

        pushings += 1
    print(pushings)
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
