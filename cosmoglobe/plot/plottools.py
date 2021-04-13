# This file contains useful functions used across different plotting scripts.
import numpy as np

def apply_logscale(m, ticks, linthresh=1):
    """
    This function converts the data to logscale, 
    and also converts the tick locations correspondingly
    """
    print("Applying semi-logscale")
    m = symlog(m,linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i,linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks

def set_style(darkmode):
    """
    This function sets the color parameter and text style
    """
    import matplotlib.pyplot as plt
    # rcParameters
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["lines.linewidth"] = 2
    plt.rc("text.latex", preamble=r"\usepackage{sfmath}",)

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

def load_cmap(cmap,logscale):
    """
    This function takes the colormap label and loads the colormap object
    """
    from .. import data as data_dir
    import matplotlib.colors as col
    from pathlib import Path

    if cmap == None: cmap = "planck"
    if "planck" in cmap:
        if "planck_log" in cmap: #logscale:
            # setup nonlinear colormap
            from matplotlib.colors import LinearSegmentedColormap
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
                    yi = (yi + 3)/10.
                    return self.cmap(yi, alpha)

                def modsinh(self, x):
                    return np.log10(0.5*(x + np.sqrt(x**2 + 4)))
            
            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap_logscale.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            cmap = col.ListedColormap(planck_cmap, "planck")
            cmap = GlogColormap(cmap)        
        else:
            cmap_path = Path(data_dir.__path__[0]) / "planck_cmap.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            if cmap.endswith("_r"):
                planck_cmap = planck_cmap[::-1]
            cmap = col.ListedColormap(planck_cmap, "planck")
    elif "wmap" in cmap:
        cmap_path = Path(data_dir.__path__[0]) / "wmap_cmap.dat"
        wmap_cmap = np.loadtxt(cmap_path) / 255.0            
        if cmap.endswith("_r"):
            planck_cmap = planck_cmap[::-1]
        cmap = col.ListedColormap(wmap_cmap, "wmap")
    elif cmap.startswith("q-"):
        import plotly.colors as pcol
        _, clab, *numvals = cmap.split("-")
        colors = getattr(pcol.qualitative, clab)
        if clab=="Plotly":
            #colors.insert(3,colors.pop(-1))
            colors.insert(0,colors.pop(-1))
            colors.insert(3,colors.pop(2))
        try:
            cmap = col.ListedColormap(colors[:int(numvals[0])], f'{clab}-{numvals[0]}')
            print("Using qualitative colormap:" + f" {clab} up to {numvals[0]}")
        except:
            cmap = col.ListedColormap(colors,clab)
            print("Using qualitative colormap:" f" {clab}")
    elif cmap.startswith("black2"):
        cmap = col.LinearSegmentedColormap.from_list(cmap,cmap.split("2"))
    else:
        try:
            import cmasher
            cmap = eval(f"cmasher.{cmap}")
        except:
            try:
                from cmcrameri import cm
                cmap = eval(f"cm.{cmap}")
            except:
                import matplotlib.pyplot as plt
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
    m = m/linthresh/(2*np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))

def autoparams(auto, sig, title, ltitle, unit, ticks, logscale, cmap):
    from pathlib import Path
    from .. import data as data_dir
    if auto is not None:
        import json
        with open(Path(data_dir.__path__[0]) /'autoparams.json', 'r') as f:
            autoparams = json.load(f)
        try:
            params = autoparams[auto]
        except:
            print(f"Component label {auto} not found. Using unidentified profile.")
            print(autoparams.keys())
            params = autoparams["unidentified"]

        #If none will be set to IQU
        params["left_title"] = ["I","Q","U"][sig]

        # If l=0 use intensity cmap and ticks, else use QU
        l = 0 if sig<1 else -1
        params["cmap"] = params["cmap"][l]
        params["ticks"] = params["ticks"][l]

        if any(j in auto for j in ["residual", "freqmap", "bpcorr", "smap"]):
            # Add number, such as frequency, to title.
            number = ''.join(filter(lambda i: i.isdigit(), auto))
            params["title"]=params["title"]+"{"+number+"}"

        if "rms" in auto:
            params["title"] += "^{\mathrm{RMS}}"
            params["cmap"] = "neutral"
            params["ticks"] = "auto"
        if "stddev" in auto:
            params["title"] = "\sigma_{\mathrm{" + param["title"] + "}}"
            params["cmap"] = "neutral"
            params["ticks"] = "auto"
        if "mean" in auto:
            params["title"] =  "\langle " + comp["title"] +"\rangle"

        # This is messy
        if title is not None: params["title"] = title
        if ltitle is not None: params["left_title"] = ltitle
        if unit is not None: params["unit"] = unit
        if ticks is not None: params["ticks"] = ticks
        if logscale is not None: params["logscale"] = logscale
        if cmap is not None: params["cmap"] = cmap
    else:
        params = {"title": title, 
                    "left_title": ["I","Q","U"][sig],
                    "unit": unit,
                    "ticks": ticks,
                    "logscale": logscale,
                    "cmap": cmap,
                }
    return params

def make_fig(figsize, fignum, hold, subplot, reuse_axes, projection=None):
    import matplotlib.pyplot as plt
    # From healpy
    nrows, ncols, idx = (1,1,1)
    width, height = figsize
    if not (hold or subplot or reuse_axes):
        fig = plt.figure(fignum, figsize=(width,height))
    elif hold:
        fig = plt.gcf()
        left, bottom, right, top = np.array(fig.gca().get_position()).ravel()
        fig.delaxes(fig.gca())
    elif reuse_axes:
        fig = plt.gcf()
    else:  # using subplot syntax
        if hasattr(subplot, "__len__"):
            nrows, ncols, idx = subplot
        else:
            nrows, ncols, idx = subplot // 100, (subplot % 100) // 10, (subplot % 10)
        
        # If figure exists, get. If not, create.
        from matplotlib import _pylab_helpers
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            fig = figManager.canvas.figure
        else:
            figsize = (width*ncols, height*nrows)
            fig = plt.figure(figsize=figsize)

        if idx < 1 or idx > ncols * nrows:
            raise ValueError("Wrong values for subplot: %d, %d, %d" % (nrows, ncols, idx))

    ax = fig.add_subplot(int(f"{nrows}{ncols}{idx}"), projection=projection)

    return fig, ax