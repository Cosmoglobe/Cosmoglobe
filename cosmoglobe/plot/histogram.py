import numpy as np
import matplotlib.pyplot as plt
from .plottools import set_style, make_fig


def hist(
    input,
    *args,
    prior=None,
    darkmode=False,
    figsize=None,
    fignum=None,
    subplot=None,
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
    prior : touple
        mean and stddev of gaussian distribution overlay
    """

    # Make figure
    fig, ax = make_fig(figsize, fignum, hold, subplot, reuse_axes, darkmode,)

    # Use matplotlib histogram function with specific options
    n, bins, patches = plt.hist(
        input, *args, histtype="step", density=True, stacked=True, **kwargs
    )

    # Overlay a gaussian prior
    if prior is not None:
        import scipy.stats as stats

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
        rotation=90, va="center",
    )
    return n, bins, patches
