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
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
):
    if isinstance(input, str) and input.endswith(".h5") and dataset is not None:
        chain = Chain(input)
        if ylabel == None:
            ylabel = dataset
        #component, *items = dataset.split("/")
        input = chain.get(dataset)

    # Make figure
    fig, ax = make_fig(
        figsize,
        fignum,
        hold,
        subplot,
        reuse_axes,
        darkmode,
    )

    if input.ndim < 2:
        input.reshape(-1, 1, 1)
    elif input.ndim == 2:
        input.reshape(0, 1, -1)
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
