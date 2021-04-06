import click
import sys
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import astropy.units as u
from astropy.units import cm, imperial
from cosmoglobe.tools.map import to_stokes, StokesMap

hp.disable_warnings()
# Fix for macos openMP duplicate bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def mollplot(
    map_,
    sig=[
        0,
    ],
    auto=False,
    min=False,
    max=False,
    mid=[],
    rng="auto",
    colorbar=False,
    graticule=False,
    lmax=None,
    fwhm=0.0,
    mask=None,
    mfill=None,
    logscale=False,
    mdmask=None,
    remove_dipole=True,
    remove_monopole=True,
    cmap=None,
    title=None,
    ltitle=None,
    outdir=".",
    outname=None,
    size="m",
    white_background=False,
    darkmode=False,
    fontsize=11,
    dpi=300,
    xsize=2000,
    hires=False,
    gif=False,
):
    """
    Plots a fits file, a h5 data structure or a data array in Mollview with
    pretty formatting. Option of autodecting component base on file-string with
    json look-up table.

    Parameters
    ----------
    map_ : cosmoglobe map object
        cosmoglobe map object with I (optional Q and U)
    sig : list of int or string ["I", "Q", "U"]
        Signal indices to be plotted. 0, 1, 2 interprated as IQU.
        Supports multiple, such that -sig 1 -sig 2.
        default = [0,]
    auto : bool
        Toggle parameter autodetector. Automatic identification of plotting
        parameters based on information from autoparams.json default = False
    min :  str
        Minimum value of colorbar. Overrides autodetector.
        default = False
    max :  str
        Maximum value of colorbar. Overrides autodetector.
        default = False
    mid : list
        Adds tick values between min and max. Accepts multiple ex. -mid 2 -mid
        3. default = []
    range : str
        Min and max value around 0 (ex. 10).
        Also compatible with "auto": sets to 97.5 percentile of data.
        Also compatible with "minmax": sets to min and max values.
        default = "auto"
    colorbar : bool
        Adds a colorbar, and "cb" to output filename.
        default = False
    graticule : bool
        Adds graticule to figure.
        default = False
    lmax : float
        Set the lmax of alm input data from hdf.
        If reading hdf and not set, this is automatically found.
        default = None
    fwhm : float
        Optional map smoothing. FWHM of gaussian smoothing in arcmin.
        default = 0.0
    mask : array of healpix mask
        Apply a mask file to data
        default = None
    mfill : str
        Color to fill masked area, for example "gray".
        Transparent by default.
        defualt : None
    mdmask : str or array
        Mask for mono-dipole removal. If "auto", uses 30 degree cutoff.
        Removes both by default. Toggle off dipole or monopole removal with
        remove_dipole or remove_monopole arguments.
        default = None
    remove_dipole : bool
        If mdmask is specified, fits and removes a dipole. 
        default = True
    remove_monopole : bool
        If mdmask is specified, fits and removes a monopole. 
        default = True
    logscale : str
        Normalizes data using a semi-logscale linear between -1 and 1.
        Autodetector uses this sometimes, you will be warned.
        default = None
    size : str
        Size of output maps. 1/3, 1/2 and full page width (8.8/12/18cm) [ex. x,
        s, m or l, or ex. slm for all]" default = "m"
    white_background : bool
        Sets white background. Transparent by default.
        default = False
    darkmode : bool
        Plots all outlines in white for dark backgrounds, and adds "dark" in
        filename.
        default = False
    png : bool
        Saves file as png as opposed to PDF. Overwritten by outname extension.
        default = False
    cmap : str
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default = None
    title : str
        Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    ltitle : str
        Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    scale : float
        Scale data by multiplicatiive factor. [ex. 1e-6 for muK to K
        default = None
    outdir : path
        Optiional different output directory.
        default = "."
    outname : str
        Output filename. If not specified, descriptive filename will be
        generated from chosen params. Extension determines filetype and
        overrides -png flag.
        default = None
    gif : bool
        Make gif is input is list of data.
        defaullt = False
    oldfont : bool
        Use DejaVu font as oppoosed to Times.
        default = False
    fontsize : int
        Fontsize
        default = 11
    dpi : int
        DPI (Dots per inch). For figure resolution.
        default = 300
    xsize : int
        Figure width in pixels. Uses xsize*(xsize/2)
        default = 2000
    hires : bool
        For high resolution figures. Sets dpi to 3000 and xsize to 10000.
        default = False
    """
    # Check if input is cosmoglobe map object
    if not isinstance(map_, (StokesMap)):
        map_ = to_stokes(map_)

    nside = map_.nside
    unit = map_.unit
    comp = map_.label
    print(f"Plotting {map_.label} nside {nside}")
    # Projection arrays
    grid_pix, longitude, latitude = project_map(
        nside,
        xsize=xsize,
        ysize=int(xsize / 2.0),
    )

    # Mask map
    if mask: 
        "Applying mask"
        map_.mask(mask)

    # Set plotting rcParams
    set_style(darkmode)
    if hires:
        xsize = 10000
        dpi = 1000


    for i, pol in enumerate(sig):
        if isinstance(pol, int):
            pol = ["I", "Q", "U"][pol]

        # Remove mono/dipole
        if mdmask:
            # If mdmask exists, remove mono and dipole. Manually toggle either off.
            map_.remove_md(mdmask, sig=i, remove_dipole=remove_dipole, remove_monopole=remove_monopole)
        elif remove_dipole or remove_monopole:
                print("To remove monopole or dipole, please provide mdmask or set mdmask=auto")

        # Put signal into map object
        m = getattr(map_, pol)
        grid_mask = m.mask[grid_pix] if mask else None
        grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)

        """
        This is how far ive gotten
        """

        # Testparameters
        ttl = lttl = unt = ""
        mn = md = mx = None
        ticks = [False, False]
        lgscale = False
        cmp = "jet"

        #### Commandline priority ####
        if logscale != None:
            lgscale = logscale
        if title:
            ttl = title
        if ltitle:
            lttl = ltitle
        if unit:
            unt = unit

        #### Colorbar ticks ####
        ticks = get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto)
        ticklabels = [fmt(i, 1) for i in ticks]

        #### Logscale ####
        if lgscale: m, ticks = apply_logscale(m, ticks, linthresh=1)
            
        #### Color map #####
        cmap_ = get_cmap(cmap, cmp, logscale=lgscale)

        # Terminal ouput
        print(f"FWHM: {fwhm}")
        print(f"nside: {nside}")
        print(f"Ticks: {ticks}")
        print(f"Unit: {unt}")
        print(f"Title: {ttl}")
        
        # Pick sizes from size dictionary
        sizes = [{"x": 7, "s": 8.8, "m": 12.0, "l": 18.0,}[x] * cm for x in size]
        # Plot different sizes
        for width in sizes:
            print(f"size: {str(width)}")
            width = width.to(imperial.inch)
            height = width / 2.0
            if colorbar:
                height *= 1.275  # Make sure text doesnt change with colorbar
            fig = plt.figure(figsize=(width.value, height.value))
            ax = fig.add_subplot(111, projection="mollweide")

            image = plt.pcolormesh(
                longitude[::-1],
                latitude,
                grid_map,
                vmin=ticks[0],
                vmax=ticks[-1],
                rasterized=True,
                cmap=cmp,
                shading="auto",
                animated=gif,
            )

            #### Graticule ####
            if graticule:
                apply_graticule(ax, width)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])  # rm lonlat ticklabs

            #### Colorbar ####
            if colorbar:
                apply_colorbar(
                    fig,
                    image,
                    ticks,
                    ticklabels,
                    unit,
                    fontsize,
                    linthresh=1,
                    logscale=lgscale,
                )

            #### Right Title ####
            plt.text(
                4.5,
                1.1,
                r"%s" % ttl,
                ha="center",
                va="center",
                fontsize=fontsize,
            )

            #### Left Title ####
            plt.text(
                -4.5,
                1.1,
                r"%s" % lttl,
                ha="center",
                va="center",
                fontsize=fontsize,
            )

    # TODO, parse header information somewhere?

def apply_colorbar(fig, image, ticks, ticklabels, unit, fontsize, linthresh, logscale):
    click.echo(click.style("Applying colorbar", fg="yellow"))
    from matplotlib.ticker import FuncFormatter
    cb = fig.colorbar(image, orientation="horizontal", shrink=0.4, pad=0.04, ticks=ticks, format=FuncFormatter(fmt),)
    #colorbar_ticks = np.array([-1e3, -1e2, -10, -1, 0, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    #colorbar_boundaries = np.concatenate([-1 * np.logspace(0, 3, 80)[::-1], np.linspace(-1, 1, 10), np.logspace(0, 7, 150)])
    #cb = fig.colorbar(image, orientation='horizontal', format=FuncFormatter(fmt), shrink=1., pad=0.05, boundaries = colorbar_boundaries, ticks=colorbar_ticks)
    #ticklabels = [fmt(i, 1) for i in colorbar_ticks]
    #ticklabels = [fmt(i, 1) for i in ticks]
    cb.ax.set_xticklabels(ticklabels)
    cb.ax.xaxis.set_label_text(unit)
    cb.ax.xaxis.label.set_size(fontsize)
    if logscale:
        linticks = np.linspace(-1, 1, 3)*linthresh
        logmin = np.round(ticks[0])
        logmax = np.round(ticks[-1])

        logticks_min = -10**np.arange(0, abs(logmin)+1)
        logticks_max = 10**np.arange(0, logmax+1)
        ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))
        #cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks

        logticks = symlog(ticks_, linthresh)
        logticks = [x for x in logticks if x not in ticks]
        cb.set_ticks(np.concatenate((ticks,logticks ))) # Set major ticks
        cb.ax.set_xticklabels(ticklabels + ['']*len(logticks))

        minorticks = np.linspace(-linthresh, linthresh, 5)
        minorticks2 = np.arange(2,10)*linthresh

        for i in range(len(logticks_min)):
            minorticks = np.concatenate((-10**i*minorticks2,minorticks))
        for i in range(len(logticks_max)):
            minorticks = np.concatenate((minorticks, 10**i*minorticks2))

        minorticks = symlog(minorticks, linthresh)
        minorticks = minorticks[ (minorticks >= ticks[0]) & ( minorticks<= ticks[-1]) ] 
        cb.ax.xaxis.set_ticks(minorticks, minor=True)

    cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize-2,)
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb

def apply_graticule(ax, width):
    click.echo(click.style("Applying graticule", fg="yellow"))
    #### Formatting ######    
    from matplotlib.projections.geo import GeoAxes
    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        #shifts labelling by pi
        #Shifts labelling from -180,180 to 0-360
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2 * np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    ax.set_longitude_grid(60)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))
    if width < 10:
        ax.set_latitude_grid(45)
        ax.set_longitude_grid_ends(90)            
        ax.grid(True)

def apply_logscale(m, ticks, linthresh=1):
    click.echo(click.style("Applying semi-logscale", fg="yellow", blink=True, bold=True))
    m = symlog(m,linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i,linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks

def set_style(darkmode):
    # rcParameters
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1

    # Preset high resolution parameters for superbig plots

    ## Set backend based on output filename or
    # plt.rcParams["backend"] = "agg" if png else "pdf"
    # if outname:
    #    plt.rcParams["backend"] = "agg" if outname.endswith("png") else "pdf"

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

def get_params(m, outfile, signal_label,):
    outfile = os.path.split(outfile)[-1] # Remove path 
    sl = signal_label.split("_")[0]
    if sl in ["Q", "U", "QU"]:
        i = 1
    elif sl == "P":
        i = 2
    else:
        i = 0
    import json
    from pathlib import Path
    with open(Path(__file__).parent /'autoparams.json', 'r') as f:
        params = json.load(f)
    for label, comp in params.items():
        if tag_lookup(comp["tags"], outfile+signal_label):
            click.echo(click.style("{:-^48}".format(f"Detected {label} signal {sl}"),fg="yellow"))
            if label in ["residual", "freqmap",  "smap", "bpcorr"]:
                from re import findall
                if label == "smap": tit = str(findall(r"tod_(.*?)_Smap", outfile)[0])
                if label == "freqmap": 
                    tit = str(findall(r"BP_(.*?)_", outfile)[0])
                if label == "bpcorr": 
                    tit = str(findall(r"tod_(.*?)_bpcorr", outfile)[0])
                if label == "residual":
                    if "WMAP" in outfile:
                        tit = str(findall(r"WMAP_(.*?)_", outfile)[0])
                    elif "Haslam" in outfile:
                        tit = "Haslam"
                    else:
                        if "res_"in outfile:
                            tit = str(findall(r"res_(.*?)_", outfile)[0])
                        else:
                            tit = str(findall(r"residual_(.*?)_", outfile)[0])
                    if "_P_" in outfile and sl != "T":
                        comp["ticks"] = [[-10, 0, 10]]
                    if "857" in outfile:
                        m *= 2.2703e-6 
                        comp["ticks"] = [[-300,0,300]]
                    elif "Haslam" in outfile:
                        comp["ticks"] = [[-1e4, 0, 1e4]]
                comp["comp"] = r"${"+tit+"}$"

            comp["comp"] = comp["comp"].lstrip('0')    
            #print(i,sl,len(comp["cmap"]),len(comp["ticks"]))
            if i>=len(comp["cmap"]): i = len(comp["cmap"])-1
            comp["cmap"] = comp["cmap"][i]
            comp["ticks"] = comp["ticks"][i]
            ttl, lttl = get_title(comp,outfile,signal_label,)
            
            if comp["ticks"] == "auto": comp["ticks"] = get_percentile(m,97.5)
            if label == "chisq": ttl = r"$\chi^2$"
            if label == "bpcorr": ttl ="$s_{\mathrm{leak}}^{"+tit.lstrip('0')+"}}$"
            if comp["unit"]: comp["unit"] = r"$"+comp["unit"].replace('$','')+"$"
            return (m,  ttl, lttl, comp["unit"], comp["ticks"], comp["cmap"], comp["logscale"],)
    # If not recognized
    click.echo(click.style("{:-^48}".format(f"Map not recognized, plotting with min and max values"),fg="yellow"))
    comp = params["unidentified"]
    comp["comp"] = signal_label.split("_")[-1]
    ttl, lttl = get_title(comp,outfile,signal_label,)
    if comp["ticks"][i] == "auto": comp["ticks"] = get_percentile(m,97.5)
    comp["cmap"] = comp["cmap"][i]
    return (m,  ttl, lttl, comp["unit"], comp["ticks"], comp["cmap"], comp["logscale"],)

def get_cmap(cmap, cmp, logscale=False):
    # Chose colormap manually
    if cmap == None:
        # If not defined autoset or goto planck
        cmap = cmp
    if "planck" in cmap:
        from pathlib import Path
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

            cmap_path = Path(__file__).parent / "planck_cmap_logscale.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            cmap = col.ListedColormap(planck_cmap, "planck")
            cmap = GlogColormap(cmap)        
        else:
            cmap_path = Path(__file__).parent / "planck_cmap.dat"
            planck_cmap = np.loadtxt(cmap_path) / 255.0            
            if cmap.endswith("_r"):
                planck_cmap = planck_cmap[::-1]
            cmap = col.ListedColormap(planck_cmap, "planck")


    elif "wmap" in cmap:
        from pathlib import Path
        cmap_path = Path(__file__).parent / "wmap_cmap.dat"
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
            click.echo(click.style("Using qualitative colormap:", fg="yellow") + f" {clab} up to {numvals[0]}")
        except:
            cmap = col.ListedColormap(colors,clab)
            click.echo(click.style("Using qualitative colormap:", fg="yellow") + f" {clab}")
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
                cmap = plt.get_cmap(cmap)

    click.echo(click.style("Colormap:", fg="green") + f" {cmap.name}")
    return cmap

def get_title(comp, outfile, signal_label,):
    sl = signal_label.split("_")[0]
    if tag_lookup(["STDDEV","_stddev"], outfile+signal_label):
        if comp["special"]:
            ttl = r"$\sigma_{\mathrm{" + comp["param"].replace("$","") + "}}$"
        else:
            ttl =   r"$\sigma_{\mathrm{" + comp["comp"] + "}}$"
        comp["cmap"] = "neutral"
        comp["ticks"] = "auto"
        comp["logscale"] = comp["special"] = False
    elif tag_lookup(["RMS","_rms",], outfile+signal_label):
        ttl =  comp["param"] + r"$_{\mathrm{" + comp["comp"] + "}}^{\mathrm{RMS}}$" 
        comp["cmap"] = "neutral"
        comp["ticks"] = "auto"
        comp["logscale"] = comp["special"] = False
    elif tag_lookup(["mean"], outfile+signal_label):
        ttl = r"$\langle $" + comp["param"] + r"$_{\mathrm{" + comp["comp"] + "}}^{ }$" + r"$\rangle$"
    elif tag_lookup(["diff"], outfile+signal_label):
        if "dx12" in outfile: difflabel = "\mathrm{2018}"
        difflabel = "\mathrm{NPIPE}" if "npipe" in outfile else ""
        ttl = r"$\Delta$ " + comp["param"] + r"$_{\mathrm{" + comp["comp"] + "}}^{" + difflabel + "}$"
    else:
        ttl =  comp["param"] + r"$_{\mathrm{" + comp["comp"] + "}}^{ }$" 
    try:
        ttl = comp["custom"]
    except:
        pass
    # Left signal label
    lttl = r"$" + sl +"$"
    if lttl == "$I$":
        lttl = "$T$"
    elif lttl == "$QU$":
        lttl= "$P$"

    ttl = r"$"+ttl.replace('$','')+"$"
    lttl = r"$"+lttl.replace('$','')+"$",
    if ttl == "$$": ttl =""
    if lttl == "$$": lttl =""
    return ttl, lttl

def get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto):

    # If min and max have been specified, set.
    if (
        rng == "auto" and not auto
    ):  # Mathew: I have changed this to give the desired behaviour for make_diff_plots, but I don't know if this breaks other functionality
        # it used to be
        # if rng == "auto" and not auto:
        click.echo(
            click.style("Setting range from 97.5th percentile of data", fg="yellow")
        )
        mn, mx = get_percentile(m, 97.5)
    elif rng == "minmax":
        click.echo(click.style("Setting range from min to max of data", fg="yellow"))
        mn = np.min(m)
        mx = np.max(m)
    else:
        try:
            if float(rng) > 0.0:
                mn = -float(rng)
                mx = float(rng)
                ticks = [False, 0.0, False]
        except:
            pass

    if min is False:
        min = mn
    else:
        min = float(min)

    if max is False:
        max = mx
    else:
        max = float(max)

    ticks[0] = min
    ticks[-1] = max
    if mid:
        ticks = [min, *mid, max]
    elif md:
        ticks = [min, *md, max]

    return [float(i) for i in ticks]

def get_percentile(m, percentile):
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
    # Extra fact of 2 ln 10 makes symlog(m) = m in linear regime
    m = m/linthresh/(2*np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))

def tag_lookup(tags, outfile):
    return any(e in outfile for e in tags)

def project_map(nside, xsize, ysize,):
    theta = np.linspace(np.pi, 0, ysize)
    phi = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))
    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)
    return grid_pix, longitude, latitude
