import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from .plottools import set_style, legend_positions, make_fig


def traceplot(
    input,
    header=None,
    labelval=False,
    xlabel=None,
    ylabel=None,
    nbins=None,
    burnin=0,
    cmap=None,
    figsize=(7, 2),
    darkmode=False,
    fignum=None,
    subplot=None,
    hold=False,
    reuse_axes=False,
):

    # Make figure
    fig, ax = make_fig(figsize, fignum, hold, subplot, reuse_axes, darkmode,)

    if input.ndim < 2:
        input.reshape(1, -1)
    N_comps, N = input.shape

    # Swap colors around
    colors = getattr(pcol.qualitative, cmap)
    cmap = mpl.colors.ListedColormap(colors)

    positions = legend_positions(input,)

    for i in range(N_comps):
        plt.plot(
            input[i], color=cmap(i), linewidth=2,
        )

        # Add the text to the right
        if header is not None:
            plt.text(
                N + N * 0.01,
                positions[i],
                rf"{header[i]}",
                color=cmap(i),
                fontweight="normal",
            )
        if labelval:
            mean = np.mean(input[i, burnin:])
            std = np.std(input[i, burnin:])
            label2 = rf"{mean:.2f}$\pm${std:.2f}"
            plt.text(
                N + N * 0.1, positions[i], label2, color=cmap(i), fontweight="normal"
            )

    ax.set_xlim(right=N)
    ax.set_ylabel(ylabel)
    # Tick and spine parameters
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="in")
    plt.yticks(
        rotation=90, va="center",
    )
    plt.subplots_adjust(wspace=0, hspace=0.0, right=1)
    plt.tight_layout()
