from e13tools import raise_error
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .plottools import set_style, make_fig, get_percentile
from cosmoglobe.h5.chain import Chain
import scipy.stats as stats


def hist(
    input,
    *args,
    dataset=None,
    prior=None,
    xlabel=None,
    ylabel=None,
    field=0,
    darkmode=False,
    figsize=None,
    sub=None,
    hold=False,
    reuse_axes=False,
    **kwargs,
):
    """
    This function is a wrapper on the existing matplotlib histogram with custom styling.
    Possible to plot accompanying normalized prior distribution

    Parameters
    ----------
    input : array
        data input array
    prior : touple, optional
        mean and stddev of gaussian distribution overlay
        default: None
    dataset : str, optional
        dataset to plot if passing hdf5 file
        default: None
    xlabel : str, optional
        label x axis
        default: None
    ylabel : str, optional
        label y axis
        default: None
    field : int, optional
        which field to plot if .fits file is passed
        default: 0
    darkmode : bool, optional
        turn all axis elements white for optimal dark visualization
        default: False
    figsize : touple, optional
        size of figure
        default: None
    sub : int, scalar or sequence, optional
        Use only a zone of the current figure (same syntax as subplot).
        Default: None
    hold : bool, optional
        If True, replace the current Axes by a MollweideAxes.
        use this if you want to have multiple maps on the same
        figure.
        Default: False
    reuse_axes : bool, optional
        If True, reuse the current Axes (should be a MollweideAxes). This is
        useful if you want to overplot with a partially transparent colormap,
        such as for plotting a line integral convolution.
        Default: False

    """
    hp.mollview
    if isinstance(input, str):
        if input.endswith(".h5") and dataset is not None:
            chain = Chain(input)
            if ylabel == None:
                ylabel = dataset
            input = chain.get(dataset)
        elif input.endswith(".fits"):
            input = hp.read_map(input, field=field)
        else:
            raise_error("Input format not recognized")

    # Make figure
    fig, ax = make_fig(
        figsize,
        None,
        hold,
        sub,
        reuse_axes,
        darkmode,
    )

    # Use matplotlib histogram function with specific options
    n, bins, patches = plt.hist(input, *args, histtype="step", density=True, stacked=True, **kwargs)
    # Overlay a gaussian prior
    if prior is not None:
        dx = bins[1] - bins[0]
        x = np.linspace(bins[0], bins[-1], len(input))
        norm = sum(n) * dx
        Pprior = stats.norm.pdf(x, prior[0], prior[1])  # *norm
        plt.plot(
            x,
            Pprior * norm,
            linestyle=":",
            label=r"$\mathcal{N}(" + f"{prior[0]},{prior[1]}" + ")$",
        )

    # Tick and spine parameters
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="in")
    plt.yticks(
        rotation=90,
        va="center",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return n, bins, patches
