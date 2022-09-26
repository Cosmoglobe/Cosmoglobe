from re import A
import time
totaltime = time.time()
import click
import sys
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from cosmoglobe.release.tools import arcmin2rad
import os
# Fix for macos openMP duplicate bug
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#print("Importtime:", (time.time() - totaltime))

def trygveplot(input, dataset=None, nside=None, auto=False, min=False, max=False, mid=[], rng="auto", colorbar=False,
            graticule=False, lmax=None, fwhm=0.0, mask=None, mfill=None, sig=[0,], remove_dipole=None, remove_monopole=None,
            logscale=False, size="m", white_background=False, darkmode=False, png=False, cmap=None, title=None,
            ltitle=None, unit=None, scale=None, outdir='.', outname=None, verbose=False, fontsize=11, gif=False,
            oldfont=False, dpi=300, xsize=2000, hires=False):
    """
    Plots a fits file, a h5 data structure or a data array in Mollview with pretty formatting.
    Option of autodecting component base on file-string with json look-up table.

    Parameters
    ----------
    input : array, str or list
        String or list of strings with filenames such as "cmb.fits" or "chain_c001.h5" for input data.
    dataset : str
        if input is a .h5 hdf-file specify which dataset to plot, for example "000007/cmb/amp_alm"
        default = None        
    nside : int
        nside for optional ud_grade (required when loading from hdf file)
        default = None
    auto : bool
        Toggle parameter autodetector.
        Automatic identification of plotting parameters based on information from autoparams.json
        default = False
    min :  str 
        Minimum value of colorbar. Overrides autodetector.
        default = False
    max :  str 
        Maximum value of colorbar. Overrides autodetector.
        default = False
    mid : list
        Adds tick values between min and max. Accepts multiple ex. -mid 2 -mid 3.
        default = []
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
    mask : str
        Apply a mask file (supply filename) to data (uses signal 0). 
        default = None
    mfill : str
        Color to fill masked area, for example "gray".
        Transparent by default.
        defualt : None
    sig : list of int
        Signal indices to be plotted. 0, 1, 2 interprated as IQU.
        Supports multiple, such that -sig 1 -sig 2.
        default = [0,]
    remove_dipole : str
        Fits and removes a dipole. Specify mask for fit, or "auto".
        default = None
    remove_monopole : str
        Fits and removes a monopole. Specify mask for fit, or "auto".
        default = None
    logscale : str
        Normalizes data using a semi-logscale linear between -1 and 1. 
        Autodetector uses this sometimes, you will be warned.
        default = None
    size : str
        Size of output maps. 
        1/3, 1/2 and full page width (8.8/12/18cm) [ex. x, s, m or l, or ex. slm for all]"
        default = "m"
    white_background : bool
        Sets white background. Transparent by default.
        default = False
    darkmode : bool
        Plots all outlines in white for dark backgrounds, and adds "dark" in filename.
        default = False
    png : bool
        Saves file as png as opposed to PDF. Overwritten by outname extension.
        default = False
    cmap : str
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher available as of now.
        Also supports qualitative plotly map, [ex. q-Plotly-4 (q for qualitative 4 for max color)]
        Sets planck as default.
        default = None
    title : str
        Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    ltitle : str
        Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    unit : str
        Sets uniit under color bar. Has LaTeX functionaliity (ex. $\mu K$".
        default = None
    scale : float
        Scale data by multiplicative factor. [ex. 1e-6 for muK to K
        default = None
    outdir : path
        Optional different output directory.
        default = "."
    outname : str
        Output filename. If not specified, descriptive filename will be generated from chosen params.
        Extension determines filetype and overrides -png flag.
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
    verbose : bool
        Verbose mode. Mainly timer outputs.
        default = False
    """
    if hires:
        xsize=10000
        dpi=1000
    if not oldfont:
        plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["backend"] = "agg" if png else "pdf"
    if outname:
        plt.rcParams["backend"] = "agg" if outname.endswith("png") else "pdf"
    plt.rcParams["axes.linewidth"] = 1
    if darkmode:
        params = ["text.color", "axes.facecolor", "axes.edgecolor", "axes.labelcolor", "xtick.color",
                  "ytick.color", "grid.color", "legend.facecolor", "legend.edgecolor"]
        for p in params:
            plt.rcParams[p] = "white"

    ### Terminal output for 
    click.echo("")
    click.echo(click.style("{:#^48}".format(""), fg="green"))
    click.echo(click.style("Plotting",fg="green") + f" {input} dset={dataset}")

    ####   READ MAP   #####
    maps_, lmax, outfile, signal_labels = get_map(input, sig, dataset, nside, lmax, fwhm,)

    # Terminal output specified signals
    click.echo(click.style("Using signals ",fg="green") + f"{sig}")
    click.echo(click.style("{:#^48}".format(""), fg="green"))

    #### Main plotting loop ####
    imgs = [] # Store images for gif
    for i, maps in enumerate(maps_):
        if maps.ndim == 1: maps = maps.reshape(1,-1)
        for pl, polt, in enumerate(sig):

            #### Select data column  #####
            m = hp.ma(maps[pl])
            signal_label = signal_labels[polt] if signal_labels else get_signallabel(polt) 
            nsid = hp.get_nside(m)
            filename=input[i] if isinstance(input, list) else input
            #### Smooth  #####
            if float(fwhm) > 0:# and filename.endswith(".fits"):
                click.echo(click.style(f"Smoothing fits map to {fwhm} arcmin fwhm",fg="yellow"))
                m = hp.smoothing(m, fwhm=arcmin2rad(fwhm), lmax=lmax,)

            #### ud_grade #####
            if nside is not None:# and filename.endswith(".fits"):
                if nsid != nside:
                    click.echo(click.style(f"UDgrading map from {nsid} to {nside}", fg="yellow"))
                    m = hp.ud_grade(m, nside,)
            else:
                nside = nsid

            #### Remove monopole or dipole #####
            if remove_dipole or remove_monopole: m = remove_md(m, remove_dipole, remove_monopole, nside)

            #### Scaling factor #####
            if scale:
                datastring = f"{outfile}{outname}"
                if "chisq" in datastring:
                    click.echo(click.style(f"Scaling chisq data with dof={scale}",fg="yellow"))
                    m = (m-scale)/np.sqrt(2*scale)
                else:
                    click.echo(click.style(f"Scaling data by {scale}",fg="yellow"))
                    m *= scale

            #### Automatic variables #####
            if auto:
                (m, ttl, lttl, unt, ticks, cmp, lgscale,) = get_params(m, outfile, outname, signal_label,)
                # Tick bug fix

                mn, md, mx= (ticks[0], None, ticks[-1])
                if not mid and len(ticks)>2:
                    if ticks[0]<ticks[1]  and ticks[-2]<ticks[-1]:
                        md = ticks[1:-1]
                    else:
                        ticks.pop(1)
            else:
                ttl = lttl = unt = ""
                mn = md = mx = None
                ticks = [False, False]
                lgscale = False
                cmp = "planck"

            #if "diff" in outname+outfile:
            #    click.echo(click.style("Changing cmap to planck because diff map!", fg="yellow", blink=True, bold=True))
            #    cmp = "planck"

            #### Commandline priority ####
            if logscale != None: lgscale = logscale
            if title: ttl = title
            if ltitle: lttl = ltitle 
            if unit: unt = unit 

            #### Colorbar ticks ####
            ticks = get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto)
            ticklabels = [fmt(i, 1) for i in ticks]
            
            # Terminal ouput
            click.echo(click.style("FWHM: ", fg="green") + f"{fwhm}")
            click.echo(click.style("nside: ", fg="green") + f"{nside}")
            click.echo(click.style("Ticks: ", fg="green") + f"{ticks}")
            click.echo(click.style("Unit: ", fg="green") + f"{unt}")
            click.echo(click.style("Title: ", fg="green") + f"{ttl}")
            
            #### Logscale ####
            if lgscale: m, ticks = apply_logscale(m, ticks, linthresh=1)
            
            #### Color map #####
            cmap_ = get_cmap(cmap, cmp, logscale=lgscale)
            
            #### Projection ####
            grid_pix, longitude, latitude = project_map(nside, xsize=xsize, ysize=int(xsize/ 2.0),)

            #### Mask ##########
            grid_map, cmap_ = apply_mask(m, mask, grid_pix, mfill, polt, cmap_) if mask else (m[grid_pix], cmap_)


            for width in get_sizes(size):
                click.echo(click.style("Size: ", fg="green") + str(width))
                #### Make figure ####
                height = width / 2.0
                if colorbar: height *= 1.275 # Make sure text doesnt change with colorbar
                if gif:
                    # Hacky gif implementation
                    if i == 0:
                        fig = plt.figure(figsize=(cm2inch(width), cm2inch(height),),)
                        ax = fig.add_subplot(111, projection="mollweide")
                else:
                    fig = plt.figure(figsize=(cm2inch(width), cm2inch(height),),)
                    ax = fig.add_subplot(111, projection="mollweide")
                
                #norm=col.SymLogNorm(linthresh=1, base=10) if logscale else None
                image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=ticks[0], vmax=ticks[-1], rasterized=True, cmap=cmap_, shading='auto', animated=gif)

                # Save img for gif
                if gif: imgs.append([image])

                #### Graticule ####
                if graticule: apply_graticule(ax, width)
                ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([]) # rm lonlat ticklabs
                
                #### Colorbar ####
                if colorbar: apply_colorbar(fig, image, ticks, ticklabels, unt, fontsize, linthresh=1, logscale=lgscale)
                
                #### Right Title ####
                plt.text(4.5, 1.1, r"%s" % ttl, ha="center", va="center", fontsize=fontsize,)
                
                #### Left Title ####
                plt.text(-4.5, 1.1, r"%s" % lttl, ha="center", va="center", fontsize=fontsize,)
                
                #### Output file ####
                plt.tight_layout()
                if gif: #output gif on last iteration only
                    if i==len(input)-1:
                        output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap_,nside,signal_label,width,outdir, gif, imgs, dpi, verbose, outname)
                else:
                    output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap_,nside,signal_label,width,outdir, gif, imgs, dpi, verbose, outname)
                    click.echo("Saved, closing fig")
                    plt.close()
                click.echo("Totaltime:", (time.time() - totaltime),) if verbose else None

def get_params(m, outfile, outname, signal_label,):
    outfile = os.path.split(outfile)[-1] # Remove path 
    outfile = f"{outfile}{outname}"
    datastring = f"{outfile}{signal_label}"
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
        if tag_lookup(comp["tags"], datastring):
            if label=="freefree" and tag_lookup(["diff"], datastring): continue
            click.echo(click.style("{:-^48}".format(f"Detected {label} signal {sl} in {datastring}"),fg="yellow"))
            if label in ["residual", "freqmap",  "smap", "bpcorr"]:
                from re import findall 
                if label == "smap": tit = str(findall(r"tod_(.*?)_Smap", outfile)[0])
                if label == "freqmap": 
                    #TODO generalize this
                    if '023-WMAP_K' in outfile:
                        tit = r'\mathit K'
                    elif '030-WMAP_Ka' in outfile:
                        tit = r'\mathit{Ka}'
                    elif '040-WMAP_Q1' in outfile:
                        tit = r'\mathit Q1'
                    elif '040-WMAP_Q2' in outfile:
                        tit = r'\mathit Q2'
                    elif '060-WMAP_V1' in outfile:
                        tit = r'\mathit V1'
                    elif '060-WMAP_V2' in outfile:
                        tit = r'\mathit V2'
                    elif '090-WMAP_W1' in outfile:
                        tit = r'\mathit W1'
                    elif '090-WMAP_W2' in outfile:
                        tit = r'\mathit W2'
                    elif '090-WMAP_W3' in outfile:
                        tit = r'\mathit W3'
                    elif '090-WMAP_W4' in outfile:
                        tit = r'\mathit W4'
                    elif "30" in outfile:
                        tit = "30"
                    elif "44" in outfile:
                        tit = "44"
                    elif "70" in outfile:
                        tit = "70"
                    elif "353" in outfile:
                        tit = "353"
                if label == "bpcorr": 
                    tit = str(findall(r"tod_(.*?)_bpcorr", outfile)[0])
                if label == "residual":
                    if "WMAP" in outfile:
                        m   *= 1e3
                        tit = str(findall(r"WMAP_(.*?)_", outfile)[0])
                        comp["ticks"] = [[-6, 0, 6]]
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

def symlog(m, linthresh=1.0):
    # Extra fact of 2 ln 10 makes symlog(m) = m in linear regime
    m = m/linthresh/(2*np.log(10))
    return np.log10(0.5 * (m + np.sqrt(4.0 + m * m)))

def apply_logscale(m, ticks, linthresh=1):
    click.echo(click.style("Applying semi-logscale", fg="yellow", blink=True, bold=True))
    m = symlog(m,linthresh)
    new_ticks = []
    for i in ticks:
        new_ticks.append(symlog(i,linthresh))

    m = np.maximum(np.minimum(m, new_ticks[-1]), new_ticks[0])
    return m, new_ticks

def get_signallabel(x):
    if x == 0:
        return "I"
    if x == 1:
        return "Q"
    if x == 2:
        return "U"
    return str(x)

def get_sizes(size):
    sizes = []
    if "x" in size:
        sizes.append(7)
    if "s" in size:
        sizes.append(8.8)
    if "m" in size:
        sizes.append(12.0)
    if "l" in size:
        sizes.append(18.0)
    return sizes

def get_percentile(m, percentile):
    vmin = np.percentile(m, 100.0 - percentile)
    vmax = np.percentile(m, percentile)

    vmin = 0.0 if abs(vmin) < 1e-5 else vmin
    vmax = 0.0 if abs(vmax) < 1e-5 else vmax
    return [vmin, vmax]

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
    datastring = f"{outfile}{signal_label}"
    if tag_lookup(["STDDEV","_stddev"],datastring):
        if comp["special"]:
            ttl = r"$\sigma_{\mathrm{" + comp["param"].replace("$","") + "}}$"
        else:
            ttl =   r"$\sigma_{\mathrm{" + comp["comp"] + "}}$"
        comp["cmap"] = "neutral"
        comp["ticks"] = "auto"
        comp["logscale"] = comp["special"] = False
    elif tag_lookup(["RMS","_rms",],datastring):
        ttl =  comp["param"] + r"$_{\mathrm{" + comp["comp"] + "}}^{\mathrm{RMS}}$" 
        comp["cmap"] = "neutral"
        comp["ticks"] = "auto"
        comp["logscale"] = comp["special"] = False
    elif tag_lookup(["mean"],datastring):
        ttl = r"$\langle $" + comp["param"] + r"$_{" + comp["comp"] + "}^{ }$" + r"$\rangle$"
    elif tag_lookup(["diff"],datastring):
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

def get_map(input, sig, dataset, nside, lmax, fwhm,):
    # Get maps array if .h5 file
    signal_labels=None
    maps=[]
    input = [input] if isinstance(input, (str, np.ndarray)) else input
    if not input: print("No input specified"); sys.exit()
    for input_ in input:
        if isinstance(input_, np.ndarray):
            if not nside: nside=hp.get_nside(input_)
            if not lmax:  lmax = 2.5*nside
            outfile = "figure"
            if input_.ndim > 1:
                maps_ = input_[sig]
            else:
                maps_ = input_

        elif input_.endswith(".h5"):
            from cosmoglobe.release.commands_hdf import h5map2fits
            from cosmoglobe.release.tools import alm2fits_tool
            # Get maps from alm data in .h5
            if dataset.endswith("alm"):
                if not nside:
                    click.echo(click.style("Specify nside for h5 files",fg="red"))
                    sys.exit()

                click.echo(click.style("Converting alms to map",fg="green"))

                (maps_, _, _, _, outfile,) = alm2fits_tool(input_, dataset, nside, lmax, fwhm, save=False,)
                maps_=maps_[sig]    
            # Get maps from map data in .h5
            elif dataset.endswith("map"):
                click.echo(click.style("Reading map from hdf",fg="green"))
                (maps_, _, _, outfile,) = h5map2fits(input_, dataset, save=False)
                maps_=maps_[sig]

            # Found no data specified kind in .h5
            else:
                click.echo(click.style("Dataset not found. Breaking.",fg="red"))
                click.echo(click.style(f"Does {input_}/{dataset} exist?",fg="red"))
                sys.exit()

        elif input_.endswith(".fits"):
            maps_, header = hp.read_map(input_, field=sig, h=True, dtype=None,)
            header = dict(header)
            signal_labels = []
            for i in range(int(header["TFIELDS"])):
                signal_label = header[f"TTYPE{i+1}"]
                if signal_label in ["TEMPERATURE", "TEMP"]:
                    signal_label = "T"
                if signal_label in ["Q-POLARISATION", "Q_POLARISATION"]:
                    signal_label = "Q"
                if signal_label in ["U-POLARISATION", "U_POLARISATION"]:
                    signal_label = "U"
                signal_labels.append(signal_label)            
            outfile = input_.replace(".fits", "")
        else:
            click.echo(click.style("Did not recognize data.",fg="red"))
            sys.exit()

        if maps_.ndim == 1: maps_ = maps_.reshape(1,-1)
        maps.append(maps_)
    return maps, lmax, outfile, signal_labels

def get_ticks(m, ticks, mn, md, mx, min, mid, max, rng, auto):
    # If min and max have been specified, set.
    if rng == "auto" and not auto: # Mathew: I have changed this to give the desired behaviour for make_diff_plots, but I don't know if this breaks other functionality
    # it used to be
    # if rng == "auto" and not auto:
        click.echo(click.style("Setting range from 97.5th percentile of data",fg="yellow"))
        mn,mx = get_percentile(m, 97.5)
    elif rng == "minmax":
        click.echo(click.style("Setting range from min to max of data",fg="yellow"))
        mn = np.min(m)
        mx = np.max(m)
    else:
        try:
            if float(rng)>0.0:
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

def apply_mask(m, mask, grid_pix, mfill, polt, cmap):
    click.echo(click.style(f"Masking using {mask}", fg="yellow"))
    # Apply mask
    hp.ma(m)
    mask_field = polt-3 if polt>2 else polt
    m.mask = np.logical_not(hp.read_map(mask, field=mask_field, dtype=None))

    # Don't know what this does, from paperplots by Zonca.
    grid_mask = m.mask[grid_pix]
    grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)

    if mfill:
        cmap.set_bad(mfill)  # color of missing pixels
        # cmap.set_under("white") # color of background, necessary if you want to use
        # using directly matplotlib instead of mollview has higher quality output

    return grid_map, cmap

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
        
def output_map(fig, outfile, png, fwhm, colorbar, mask, remove_dipole, darkmode, white_background,cmap,nside,signal_label,width,outdir,gif,imgs,dpi, verbose, outname):
    #### filename ##
    outfile = outfile.replace("_IQU_", "_")
    outfile = outfile.replace("_I_", "_")
    filename = []
    filename.append(f"{str(int(fwhm))}arcmin") if float(fwhm) > 0 else None
    filename.append("cb") if colorbar else None
    filename.append("masked") if mask else None
    filename.append("nodip") if remove_dipole else None
    filename.append("dark") if darkmode else None
    filename.append(f"c-{cmap.name}")

    nside_tag = "_n" + str(int(nside))
    if nside_tag in outfile:
        outfile = outfile.replace(nside_tag, "")

    fn = outfile + f"_{signal_label}_w{str(int(width))}" + nside_tag

    for i in filename:
        fn += f"_{i}"


    filetype = "png" if png else "pdf"
    if outname:
        filetype = "png" if outname.endswith(".png") else "pdf"

    fn += f".{filetype}"
    if outname: fn = outname
    starttime = time.time()
    if outdir:
        fn = outdir + "/" + os.path.split(fn)[-1]

    tp = False if white_background else True  
    if gif:
        click.echo(click.style("Outputting GIF:", fg="green") + f" {fn}")
        import matplotlib.animation as animation
        interval = 500 #100
        ani = animation.ArtistAnimation(fig, imgs, interval=interval, blit=True, repeat_delay=1000)
        ani.save(fn.replace(filetype,"gif"),dpi=dpi)
    else:
        click.echo(click.style("Outputting file", fg="green") + f" {fn}")
        fig.savefig(fn, bbox_inches="tight", pad_inches=0.02, transparent=tp, format=filetype, dpi=dpi)
    click.echo("Savefig", (time.time() - starttime),) if verbose else None

def fmt(x, pos):
    """
    Format color bar labels
    """
    if abs(x) > 1e4 or (abs(x) <= 1e-3 and abs(x) > 0):
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        if float(a) == 1.00:
            return r"$10^{"+str(b)+"}$"
        elif float(a) == -1.00:
            return r"$-10^{"+str(b)+"}$"
        else:
            return fr"${a} \cdot 10^{b}$"
    elif abs(x) > 1e1 or (float(abs(x))).is_integer():
        return fr"${int(x):d}$"
    else:
        x = round(x, 2)
        return fr"${x}$"

def cm2inch(cm):
    return cm * 0.393701

def tag_lookup(tags, outfile):
    return any(e.lower() in outfile.lower() for e in tags)

def project_map(nside, xsize, ysize,):
    theta = np.linspace(np.pi, 0, ysize)
    phi = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))
    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)

    return grid_pix, longitude, latitude

def remove_md(m, remove_dipole, remove_monopole, nside):        
    if remove_monopole:
        dip_mask_name = remove_monopole
    if remove_dipole:
        dip_mask_name = remove_dipole
    # Mask map for dipole estimation
    if dip_mask_name == 'auto':
        mono, dip = hp.fit_dipole(m, gal_cut=30)
    else:
        m_masked = hp.ma(m)
        m_masked.mask = np.logical_not(hp.read_map(dip_mask_name,dtype=None,))

        # Fit dipole to masked map
        mono, dip = hp.fit_dipole(m_masked)

    # Subtract dipole map from data
    if remove_dipole:
        click.echo(click.style("Removing dipole:", fg="yellow"))
        click.echo(click.style("Dipole vector:",fg="green") + f" {dip}")
        click.echo(click.style("Dipole amplitude:",fg="green") + f" {np.sqrt(np.sum(dip ** 2))}")

        # Create dipole template
        nside = int(nside)
        ray = range(hp.nside2npix(nside))
        vecs = hp.pix2vec(nside, ray)
        dipole = np.dot(dip, vecs)
        
        m = m - dipole
    if remove_monopole:
        click.echo(click.style("Removing monopole:", fg="yellow"))
        click.echo(click.style("Mono:",fg="green") + f" {mono}")
        m = m - mono
    return m

"""

def apply_colorbar(fig, image, ticks, unit, fontsize, norm):
    click.echo(click.style("Applying colorbar", fg="yellow"))
    from matplotlib.ticker import FuncFormatter
    cb = fig.colorbar(image, orientation="horizontal", shrink=0.4, pad=0.04, format=FuncFormatter(fmt),)
    ticklabels =  [fmt(i,1) for i in ticks]
    print(cb.ax.get_xticklabels())
    click.echo(click.style("Applying colorbar", fg="yellow"))
    from matplotlib.ticker import FuncFormatter, LogLocator
    cb = fig.colorbar(image, orientation="horizontal", shrink=0.4, pad=0.04, ticks=ticks, format=FuncFormatter(fmt),)

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

    cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize,)
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb
"""
