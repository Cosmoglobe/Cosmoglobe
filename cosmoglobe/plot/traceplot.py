import matplotlib.pyplot as plt
import plotly.colors as pcol
import numpy as np
import matplotlib as mpl
from .plottools import set_style, legend_positions, make_fig

def traceplot(input, header=None, labelval=False, xlabel=None, ylabel=None, nbins=None, burnin=0,  cmap="Plotly", figsize=(12,4), darkmode=False, fignum=None, subplot=None, hold=False, reuse_axes=False,):


     # Set plotting rcParams
    set_style(darkmode)

    if input.ndim < 2:
        input.reshape(1, -1)
    N_comps, N = input.shape        

    fig, ax = make_fig(figsize, fignum, hold, subplot, reuse_axes,)

    # Swap colors around
    colors=getattr(pcol.qualitative,cmap)    
    cmap = mpl.colors.ListedColormap(colors)

    
    positions = legend_positions(input,)

    for i in range(N_comps):
        plt.plot(input[i], color=cmap(i), linewidth=2,)

        # Add the text to the right
        if header is not None:
            plt.text(
                N+N*0.01,
                positions[i], rf"{header[i]}", fontsize=12,
                color=cmap(i), fontweight='normal'
            )
        if labelval:
            mean = np.mean(input[i, burnin:])
            std = np.std(input[i, burnin:])
            label2 = rf'{mean:.2f}$\pm${std:.2f}'
            plt.text(
                N+N*0.1,
                positions[i], label2, fontsize=12,
                color=cmap(i), fontweight='normal'
            )

    plt.gca().set_xlim(right=N)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle=":")
    plt.yticks(rotation=90, va="center",)
    plt.subplots_adjust(wspace=0, hspace=0.0, right=1)
    plt.tight_layout()
