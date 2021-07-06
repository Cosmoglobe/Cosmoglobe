def hist(input, *args, prior=None, **kwargs):
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

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.colors as pcol
    from cycler import cycler

    # Style setup
    colors=getattr(pcol.qualitative,"Plotly")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.2})
    sns.set_style("whitegrid")
    plt.rcParams["mathtext.fontset"] = "stix"
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'axes.linewidth': 1.5,
        'axes.prop_cycle': cycler(color=colors),
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': 'times',
    }
    sns.set_style(custom_style)
    fontsize = 14
    plt.yticks(rotation=90, va="center", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    sns.despine(top=True, right=True, left=True, bottom=True)

    # Use matplotlib histogram function with specific options
    n, bins, patches = plt.hist(input, *args, histtype='step', density=True, stacked=True, **kwargs)

    # Overlay a gaussian prior
    if prior is not None:
        import scipy.stats as stats
        dx = bins[1]-bins[0]
        x = np.linspace(bins[0],bins[-1],len(input))
        norm = sum(n)*dx
        Pprior = stats.norm.pdf(x, prior[0], prior[1])#*norm
        plt.plot(x, Pprior*norm, linestyle=":", label=r"$\mathcal{N}("+f"{prior[0]},{prior[1]}"+")$")

    return n, bins, patches