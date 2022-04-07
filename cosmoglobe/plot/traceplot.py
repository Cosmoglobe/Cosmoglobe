import matplotlib.pyplot as plt
import numpy as np

from .plottools import *
from cosmoglobe.h5.chain import Chain


def trace(
    input,
    dataset=None,
    sig=0,
    labels=None,
    showval=True,
    burnin=0,
    xlabel=None,
    ylabel=None,
    nbins=None,
    cmap="tab10",
    figsize=(8, 3),
    darkmode=False,
    sub=None,
    hold=False,
    reuse_axes=False,
):
    """
    Make a trace plot of a quantity over gibbs samples

    Parameters
    ----------
    input : ndarray, fits file path or cosmoglobe model object
        Map data input given as numpy array either 1d or index given by 'sig'.
        Also supports fits-file path string or cosmoglobe model.
        If cosmoglobe object is passed such as 'model', specify comp or freq.
    dataset : str, optional
        if passing hdf5 file, specify dataset
        default: None
    sig : str or int, optional
        Specify which signal to plot if ndim>1.
        default: None
    labels : list of str, optional
        List of strings to use as labels for each of the components
        default: None
    showval : bool, optional
        display the mean and stddev of the distribution next to label
        default: True
    burnin : int, optional
        Number of burnin samples, used in calculating mean and stddev values
        default: 0
    xlabel : str, optional
        label x axis
        default: None
    ylabel : str, optional
        label y axis
        default: None
    nbins : int, optional,
        number of bins
        default: None
    cmap : str, optional
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default: None
    figsize : touple, optional
        size of figure
        default: None
    darkmode : bool, optional
        turn all axis elements white for optimal dark visualization
        default: False
    sub : int, scalar or sequence, optional
        Use only a zone of the current figure (same syntax as subbplot).
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
    chain = None
    if isinstance(input, str):
        if input.endswith(".h5") and dataset is not None:
            chain = Chain(input)
            if ylabel == None:
                ylabel = dataset
            # component, *items = dataset.split("/")

            input = chain.get(dataset)

    # Make figure
    fig, ax = make_fig(
        figsize,
        None,
        hold,
        sub,
        reuse_axes,
        darkmode,
    )

    if input.ndim < 2:
        input.reshape(-1, 1, 1)
    elif chain is None and input.ndim == 2:
        input = input[:, np.newaxis, :]
    elif input.ndim == 2:
        input = input[:, :, np.newaxis]

    Nsamp, Nsig, Ncomp = input.shape

    cmap = load_cmap(cmap)
    positions = legend_positions(
        input[:, sig, :],
    )

    for i in range(Ncomp):
        plt.plot(
            input[:, sig, i],
            color=cmap(i),
            linewidth=2,
        )

        # Add the text to the right
        if labels is not None:
            hpos = Nsamp * 1.01
            plt.text(
                hpos,
                positions[i],
                rf"{labels[i]}",
                color=cmap(i),
                fontweight="normal",
            )

        if showval:
            mean = np.mean(input[burnin:, sig, i])
            std = np.std(input[burnin:, sig, i])
            label2 = rf"{mean:.2f}$\pm${std:.2f}"
            valpos = Nsamp * 1.01 if labels is None else Nsamp * 1.1
            plt.text(valpos, positions[i], label2, color=cmap(i), fontweight="normal")

    ax.set_xlim(right=Nsamp)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Tick and spine parameters
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="in")
    plt.yticks(
        rotation=90,
        va="center",
    )
    plt.subplots_adjust(wspace=0, hspace=0.0, right=1)
    plt.tight_layout()
