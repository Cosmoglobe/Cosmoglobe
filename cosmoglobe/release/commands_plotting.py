import time
import os
import numpy as np
import sys
import click
from cosmoglobe.release.tools import *
import cosmoglobe.plot


@click.group()
def commands_plotting():
    pass

@commands_plotting.command()
@click.argument("input", type=click.STRING)
@click.option("-cmap", default="Plotly", help="set colormap from plotly qualitative colors", type=click.STRING)
@click.option("-long", is_flag=True, help="Full page version")
@click.option("-lambdacdm", is_flag=True, help="overplot lambdacdm")
@click.option("-min", "min_", type=click.FLOAT, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", "max_", type=click.FLOAT, help="Max value of colorbar, overrides autodetector.",)
@click.option("-lmax", default=200, type=click.INT, help="Max value of colorbar, overrides autodetector.",)
def specplot(input,cmap,long,lambdacdm,min_,max_, lmax):
    """
    Plots the file output by the Crosspec function.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.colors as pcol
    from cycler import cycler

    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style("whitegrid")

    # Swap colors around
    colors=getattr(pcol.qualitative,cmap)    
    if cmap=="Plotly":
        colors.insert(3,colors.pop(-1))
    plt.rcParams["mathtext.fontset"] = "stix"
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'font.size': 20, 
        'axes.linewidth': 1.5,
        'axes.prop_cycle': cycler(color=colors),
        'text.usetex': True,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': 'Times',
    }
    sns.set_style(custom_style)
    ell, ee, bb, eb = np.loadtxt(input, usecols=(0,2,3,6), skiprows=3, max_rows=lmax, unpack=True)

    ee = ee*(ell*(ell+1)/(2*np.pi))
    bb = bb*(ell*(ell+1)/(2*np.pi))
    eb = eb*(ell*(ell+1)/(2*np.pi))
    ratio = False
    if long:
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        if ratio:
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            sns.despine(top=True, right=True, left=False, bottom=False, ax=ax1,)
            sns.despine(top=True, right=True, left=False, bottom=True, ax=ax2,)
            l3 = ax2.semilogx(ell, bb/ee, linewidth=2, label="BB/EE", color='C2')
            ax2.set_ylabel(r"BB/EE",)
            ax2.set_xlabel(r'Multipole moment, $l$',)

        else:
            fig = plt.figure(figsize=(12,8))
            ax2 = plt.gca()

            ax1 = ax2
    

    if lambdacdm:
        import camb
        import healpy as hp
        #Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.07)
        pars.set_for_lmax(lmax+1,  lens_potential_accuracy=0)
        pars.WantTensors = True
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(params=pars, lmax=lmax+1, CMB_unit='muK', raw_cl=True,)

        TT_lcdm=powers['unlensed_scalar'][2:,0]*(ell*(ell+1)/(2*np.pi))
        EE_lcdm=powers['unlensed_scalar'][2:,1]*(ell*(ell+1)/(2*np.pi))
        TE_lcdm=powers['unlensed_scalar'][2:,3]*(ell*(ell+1)/(2*np.pi))
        from sklearn.linear_model import LinearRegression

        idx = 50
        x = TT_lcdm[10:idx].reshape(-1,1)
        eebb = (ee+bb)/2
        reg = LinearRegression().fit(x,eebb[10:idx])
        scaling_TT = [1e-6] #-1*reg.coef_

        x = EE_lcdm[10:idx].reshape(-1,1)
        reg = LinearRegression().fit(x,eebb[10:idx])
        scaling_EE = [1e-1] #-1*reg.coef_

        scaling_TE = [1e-3] #-1*reg.coef_
        from cosmoglobe.release.spectrum import fmt
        ax1.loglog(ell,scaling_TT[0]*TT_lcdm, linewidth=2, ls=":", label=r"TT $\lambda$CDM"+f" - scaled by "+fmt(scaling_TT[0],1))
        ax1.loglog(ell,scaling_EE[0]*EE_lcdm, linewidth=2, ls=":", label=r"EE $\lambda$CDM"+f" - scaled by "+fmt(scaling_EE[0],1))
        ax1.loglog(ell,scaling_TE[0]*np.abs(TE_lcdm), linewidth=2, ls=":", label=r"TE $\lambda$CDM"+f" - scaled by "+fmt(scaling_TE[0],1))
        ax1.loglog(ell,abs(scaling_TE[0]*TE_lcdm+scaling_TT[0]*TT_lcdm), linewidth=2, ls=":", label=r"TT+TE $\lambda$CDM")
        #BB_lcdm = ax1.loglog(ell, powers['tensor'][2:,2]*(ell*(ell+1)/(2*np.pi)), linewidth=2, ls=":", label=r"BB $\lambda CDM$")


    l1 = ax1.loglog(ell, ee, linewidth=2, label="EE", color='C0')
    l2 = ax1.loglog(ell, bb, linewidth=2, label="BB", color='C1')
    ax1.set_ylabel(r"$D_l$ [$\mu K^2$]",)

    #plt.semilogx(ell, eb, label="EB")


    color = "black"
    for ax in [ax1, ax2]:
        for i in ["left", "right", "top", "bottom"]:
            ax.spines[i].set_edgecolor("black")

    #plt.xlim(0,200)
    ax1.set_ylim(min_,max_)
    ax1.grid(b=True, which='minor', axis="x")
    ax2.grid(b=True, which='minor', axis="x")
    import matplotlib.ticker as mticker
    #ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    #ax2.set_xticklabels(["20",])
    ax2.set_xticks([2,5,10,20,50,100,200])#,200,500,1000])
    ax1.tick_params(axis='y', labelrotation=90) 
    ax1.set_yticklabels(ax1.get_yticks(), va="center")
    ax2.tick_params(axis='y', labelrotation=90) 
    ax2.set_yticklabels(ax2.get_yticks(), va="center")

    plt.minorticks_on()
    ax1.legend(frameon=False,)
    #ax1.legend(frameon=False)
    outname = input.replace(".dat",".pdf")

    plt.tight_layout(h_pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02,)
    plt.show()

@commands_plotting.command()
@click.argument("input", nargs=-1,)
@click.option("-dataset", type=click.STRING, help="for .h5 plotting (ex. 000007/cmb/amp_alm)")
@click.option("-nside", type=click.INT, help="nside for optional ud_grade.",)
@click.option("-auto", is_flag=True, help="Automatically sets all plotting parameters.",)
@click.option("-min", default=None, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", default=None, help="Max value of colorbar, overrides autodetector.",)
@click.option("-mid", multiple=True, help='Adds tick values "-mid 2 -mid 4"',)
@click.option("-range", default="auto", type=click.STRING, help='Color range. "-range auto" sets to 97.5 percentile of data., or "minmax" which sets to data min and max values.',)  # str until changed to float
@click.option("-colorbar", "-bar", is_flag=True, help='Adds colorbar ("cb" in filename)',)
@click.option("-graticule", is_flag=True, help='Adds graticule',)
@click.option("-lmax", default=None, type=click.FLOAT, help="This is automatically set from the h5 file. Only available for alm inputs.",)
@click.option("-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing, in arcmin.",)
@click.option("-mask", default=None, type=click.STRING, help="Masks input with specified maskfile.",)
@click.option("-mfill", default=None, type=click.STRING, help='Color to fill masked area. for example "gray". Transparent by default.',)
@click.option("-sig", default=[0,], type=click.INT, multiple=True, help="Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)",)
@click.option("-remove_dipole", default=None, type=click.STRING, help="Fits a dipole to the map and removes it. Specify mask or type auto.",)
@click.option("-remove_monopole", default=None, type=click.STRING, help="Fits a monopole to the map and removes it.",)
@click.option("-log/-no-log", "logscale", default=None, help="Plots using planck semi-logscale (Linear between -1,1). Autodetector sometimes uses this.",)
@click.option("-size", default="m", type=click.STRING, help="Size: 1/3, 1/2 and full page width (8.8/12/18cm) [ex. x, s, m or l, or ex. slm for all], m by default",)
@click.option("-white_background", is_flag=True, help="Sets the background to be white. (Transparent by default [recommended])",)
@click.option("-darkmode", is_flag=True, help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',)
@click.option("-png", is_flag=True, help="Saves output as .png ().pdf by default)",)
@click.option("-cmap", default=None, type=click.STRING, help="Chose colormap (ex. sunburst, planck, etc). Available are matplotlib and cmasher. Also qualitative plotly [ex. q-Plotly-4 (q for qualitative 4 for max color)]",)
@click.option("-title", default=None, type=click.STRING, help="Set title (Upper right), has LaTeX functionality. Ex. $A_{s}$.",)
@click.option("-ltitle", default=None, type=click.STRING, help="Set title (Upper left), has LaTeX functionality. Ex. $A_{s}$.",)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-scale", default=None, type=click.FLOAT, help="Scale input map [ex. 1e-6 for muK to K]",)
@click.option("-outdir", type=click.Path(exists=True), help="Output directory for plot",)
@click.option("-outname", type=click.STRING, help="Output filename, overwrites autonaming",)
@click.option("-gif", is_flag=True, help="Make gifs from input",)
@click.option("-oldfont", is_flag=True, help="Use the old DejaVu font and not Times",)
@click.option("-fontsize", default=11, type=click.INT, help="Fontsize",)
@click.option("-dpi", default=300, type=click.INT, help="DPI",)
@click.option("-xsize", default=2000, type=click.INT, help="figuresize in px (2000 default)",)
@click.option("-hires", is_flag=True, help="sets dpi to 3000 and xsize to 10000",)
@click.option("-verbose", is_flag=True, help="Verbose mode")
def plot(input, dataset, nside, auto, min, max, mid, range, colorbar, graticule, 
        lmax, fwhm, mask, mfill, sig, remove_dipole, remove_monopole, logscale, 
        size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, 
        outdir, outname, gif, oldfont, fontsize, dpi, xsize, hires, verbose,):
    """
    Plots map from .fits or h5 file.
    ex. c3pp plot coolmap.fits -bar -auto -lmax 60 -darkmode -pdf -title $\beta_s$
    ex. c3pp plot coolhdf.h5 -dataset 000007/cmb/amp_alm -nside 512 -remove_dipole maskfile.fits -cmap cmasher.arctic 

    Uses 97.5 percentile values for min and max by default!\n
    RECOMMENDED: Use -auto to autodetect map type and set parameters.\n
    Some autodetected maps use logscale, you will be warned.
    """
    if min is None: min = False
    if max is None: max = False
    from cosmoglobe.release.plotter import trygveplot
    trygveplot(input, dataset, nside, auto, min, max, mid, range, colorbar, graticule,
                lmax, fwhm, mask, mfill, sig, remove_dipole, remove_monopole, logscale, 
                size, white_background, darkmode, png, cmap, title, ltitle, unit, scale, 
                outdir, outname, verbose, fontsize, gif, oldfont, dpi, xsize, hires)
    # Mid and range are replaced with ticks in the new version. Also, max is
    # overwritten. But most importantly, we can just use the comp flag...
    # I would do a for loop over all the sigs, and save each one after calling
    # cosmoglobe.plot
    if hires:
        xsize=10000
        dpi=1000
    #cosmoglobe.plot(input, dataset, sig=sig[i], comp=comp, freq=freq,
    #    xsize=xsize, darkmode=darkmode)


@commands_plotting.command()
@click.argument("filename", type=click.STRING)
@click.option("-lon", default=0,type=click.INT)
@click.option("-lat", default=0,type=click.INT)
@click.option("-size", default=20, type=click.INT)
@click.option("-sig", default=0, help="Which sky signal to plot",)
@click.option("-min", "min_", type=click.FLOAT, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", "max_", type=click.FLOAT, help="Max value of colorbar, overrides autodetector.",)
@click.option("-range", "rng", type=click.FLOAT, help="Color bar range")
@click.option("-title", default=None, type=click.STRING, help="Set title, has LaTeX functionality. Ex. $\mu$",)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-cmap", default="planck", help="Choose different color map (string), such as Jet or planck",)
@click.option("-graticule", is_flag=True, help="Add graticule",)
@click.option("-log", is_flag=True, help="Add graticule",)
@click.option("-nobar", is_flag=True, help="remove colorbar",)
@click.option("-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing, in arcmin.",)
@click.option("-remove_dipole", default=None, type=click.STRING, help="Fits a dipole to the map and removes it. Specify mask or type auto.",)
@click.option("-remove_monopole", default=None, type=click.STRING, help="Fits a monopole to the map and removes it.",)
@click.option("-outname", help="Output filename, else, filename with different format.",)
def gnomplot(filename, lon, lat, sig, size, min_, max_, rng, title, unit, cmap, graticule, log, nobar, fwhm, remove_dipole, remove_monopole, outname):
    """
    Gnomonic view plotting. 
    """
    import healpy as hp
    import matplotlib.pyplot as plt
    from cosmoglobe.release.plotter import fmt, remove_md
    from functools import partial
    
    plt.rcParams["backend"] = "pdf"
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rc('font', family='serif',)
    plt.rc("text.latex", preamble=r"\usepackage{sfmath}",)


    if cmap == "planck":
        import matplotlib.colors as col
        from pathlib import Path
        if log:
            cmap = Path(__file__).parent / "planck_cmap_logscale.dat"
        else:
            cmap = Path(__file__).parent / "planck_cmap.dat"
        cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0, "planck")
    else:
        try:
            import cmasher
            cmap = eval(f"cmasher.{cmap}")
        except:
            cmap = plt.get_cmap(cmap)

    xsize = 5000
    reso = size*60/xsize
    fontsize=10
    x = hp.read_map(filename, field=sig, verbose=False, dtype=None)
    nside=hp.get_nside(x)
        
    if float(fwhm) > 0:
        click.echo(click.style(f"Smoothing fits map to {fwhm} arcmin fwhm",fg="yellow"))
        x = hp.smoothing(x, fwhm=arcmin2rad(fwhm), lmax=3*nside,)
    if remove_dipole or remove_monopole: x = remove_md(x, remove_dipole, remove_monopole, nside)

    proj = hp.projector.GnomonicProj(rot=[lon,lat,0.0], coord='G', xsize=xsize, ysize=xsize, reso=reso)
    reproj_im = proj.projmap(x, vec2pix_func = partial(hp.vec2pix, nside))

    if rng:
        min_ = -rng
        max_ = rng
    #norm="log" if log else None
    fig, ax = plt.subplots()
    image = plt.imshow(reproj_im, origin='lower', interpolation='nearest', vmin=min_, vmax=max_, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.text(0.02,0.98,title, color="black", va="top", weight='bold', transform=ax.transAxes)
    if not nobar:
        # colorbar
        from matplotlib.ticker import FuncFormatter
        cb = plt.colorbar(image, orientation="horizontal", shrink=0.5, pad=0.03, format=FuncFormatter(fmt))
        cb.ax.tick_params(which="both", axis="x", direction="in", labelsize=fontsize)
        cb.ax.xaxis.set_label_text(unit)
        cb.ax.xaxis.label.set_size(fontsize+2)

    if graticule:
        hp.graticule()
    if not outname:
        outname = filename.replace(".fits", f"_gnomonic_{lon}lon{lat}lat_{size}x{size}deg.pdf")

    click.echo(f"Outputting {outname}")
    plt.savefig(outname, bbox_inches="tight", pad_inches=0.02, transparent=True, format="pdf",)


@commands_plotting.command()
@click.argument("filename", type=click.STRING)
@click.option("-size", default=20, type=click.INT)
@click.option("-sig", default=0, help="Which sky signal to plot",)
@click.option("-min", "min_", type=click.FLOAT, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", "max_", type=click.FLOAT, help="Max value of colorbar, overrides autodetector.",)
@click.option("-range", "rng", help="Color bar range")
@click.option("-title", default=None, type=click.STRING, help="Set title, has LaTeX functionality. Ex. $\mu$",)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-cmap", default="planck", help="Choose different color map (string), such as Jet or planck",)
@click.option("-graticule", is_flag=True, help="Add graticule",)
@click.option("-log", is_flag=True, help="Add graticule",)
@click.option("-bar", is_flag=True, help="remove colorbar",)
@click.option("-outname", help="Output filename, else, filename with different format.",)
def cartplot(filename, sig, size, min_, max_, rng, title, unit, cmap, graticule, log, bar, outname):
    """
    Cartesian view plotting. 
    """
    #c3pp cartplot BP_dust_IQU_n1024_v1.fits -sig 0 -min 10 -max 1000 -cmap sunburst -nobar -log
    #c3pp cartplot BP_dust_IQU_n1024_v1.fits -sig 2 -range 50 -cmap iceburn -nobar
    import healpy as hp
    import matplotlib.pyplot as plt
    from cosmoglobe.release.plotter import fmt, apply_logscale, get_percentile
    from functools import partial
    
    from matplotlib import rcParams, rc
    rcParams["backend"] = "pdf"
    rcParams["legend.fancybox"] = True
    rcParams["lines.linewidth"] = 2
    rcParams["savefig.dpi"] = 300
    rcParams["axes.linewidth"] = 1
    rc("text.latex", preamble=r"\usepackage{sfmath}",)

    if cmap == "planck":
        import matplotlib.colors as col
        from pathlib import Path
        if log:
            cmap = Path(__file__).parent / "planck_cmap_logscale.dat"
        else:
            cmap = Path(__file__).parent / "planck_cmap.dat"
        cmap = col.ListedColormap(np.loadtxt(cmap) / 255.0, "planck")
    else:
        try:
            import cmasher
            cmap = eval(f"cmasher.{cmap}")
        except:
            cmap = plt.get_cmap(cmap)

            
    fontsize=10
    x = hp.read_map(filename, field=sig, verbose=False, dtype=None)
    nside=hp.get_nside(x)

    if rng:
        if rng == "auto":
            min_, max_ = get_percentile(x, 97.5)
        else:
            min_ = -float(rng)
            max_ = float(rng)
    
    if log:
        x, new_ticks = apply_logscale(x, [min_, max_], 1)
        min_, max_ = new_ticks

    dpi = 1000
    xsize = 10000#3840
    ysize = xsize/2 #2400 #2160
    hp.cartview(map=x, xsize=xsize, ysize=ysize, title=title, min=min_, max=max_, cbar=bar, cmap=cmap, return_projected_map=False)
    #proj = hp.projector.CartesianProj(rot=[lon,lat,0.0], coord='G', xsize=xsize, ysize=xsize,reso=reso)
    if not outname:
        outname = filename.replace(".fits", f"_sig{sig}_cartesian.png")
    click.echo(f"Outputting {outname}")
    plt.savefig(outname, bbox_inches="tight", pad_inches=0.0, transparent=True, format="png", dpi=dpi)


@commands_plotting.command()
@click.argument("procver", type=click.STRING)
@click.option("-mask", type=click.Path(exists=True), help="Mask for calculating cmb",)
@click.option("-defaultmask", is_flag=True, help="Use default dx12 mask",)
@click.option("-freqmaps", is_flag=True, help=" Plot freqmaps",)
@click.option("-cmb", is_flag=True, help=" Plot cmb",)
@click.option("-cmbresamp", is_flag=True, help=" Plot cmbresamp",)
@click.option("-synch", is_flag=True, help=" Plot synch",)
@click.option("-ame", is_flag=True, help="Plot ame",)
@click.option("-ff", is_flag=True, help="Plot ff",)
@click.option("-dust", is_flag=True, help=" Plot dust",)
@click.option("-diff", is_flag=True, help="Creates diff maps to dx12 and npipe")
@click.option("-diffcmb", is_flag=True, help="Creates diff maps with cmb maps")
@click.option("-goodness", is_flag=True, help="Plots chisq and residuals")
@click.option("-goodness_temp", is_flag=True, help="Plots chisq and residuals")
@click.option("-goodness_pol", is_flag=True, help="Plots chisq and residuals")
@click.option("-chisq", is_flag=True, help="Plots chisq and residuals")
@click.option("-spec", is_flag=True, help="Creates emission plot")
@click.option("-all", "all_", is_flag=True, help="Plot all")
@click.pass_context
def plotrelease(ctx, procver, mask, defaultmask, freqmaps, cmb, cmbresamp, synch, ame, ff, dust, diff, diffcmb, goodness, goodness_temp, goodness_pol, chisq, spec, all_):
    """
    Plots all release files.
    """
    import os
    if not os.path.exists("figs"):
        os.mkdir("figs")

    if all_:
        freqmaps = not freqmaps; cmb = not cmb; synch = not synch; ame = not ame;
        ff = not ff; dust = not dust; diff = not diff; diffcmb = not diffcmb; spec = not spec
        goodness = not goodness; goodness_temp = not goodness_temp; goodness_pol = not goodness_pol
        chisq = not chisq

        defaultmask = True if not mask else False

    if goodness_temp or goodness_pol or chisq:
        goodness = True
    elif goodness:
        goodness_temp = goodness_pol = chisq = True




    size = "mls"
    for colorbar in [True, False]:
        if (cmbresamp and mask) or (cmbresamp and defaultmask):
            """
            Plot infilled CMB maps using the last sample of the chain
            """
            outdir = "figs/cmb/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if defaultmask:
                mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"

            try:
                # CMB I with dip
                ctx.invoke(plot, input=f"BP_c0001_Tresamp_{procver}.h5", dataset = "cmb/amp_alm", nside=1024, fwhm=14, lmax=2000, size=size, outdir=outdir, colorbar=colorbar, auto=True, range=3400)
                # CMB I without dip
                ctx.invoke(plot, input=f"BP_c0001_Tresamp_{procver}.h5", dataset = "cmb/amp_alm", nside=1024, fwhm=14, lmax=2000, size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,)
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")
            
            try:
                ctx.invoke(plot, input=f"BP_c0001_Presamp_{procver}.h5", dataset = "cmb/amp_alm", nside=1024,  fwhm=14, lmax=2000, size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,1])
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")
    

        if (cmb and mask) or (cmb and defaultmask):
            """
            Plot average CMB maps
            """

            outdir = "figs/cmb/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if defaultmask:
                mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"
                
            try:
                # CMB I with dip
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True,  range=3400)
            
                # CMB I no dip
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask, )
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,  fwhm=np.sqrt(60.0**2-14**2),)
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask,  fwhm=np.sqrt(420.0**2-14**2),range=150)

                # CMB QU at 14 arcmin, 1 degree and 7 degree smoothing
                for i, fwhm in enumerate([0.0, np.sqrt(60.0**2-14**2), np.sqrt(420.0**2-14**2)]):
                    rng = 5 if i == 2 else None
                    ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,], fwhm=fwhm, range=rng)

                # RMS maps
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,], min=0, max=30)
                ctx.invoke(plot, input=f"BP_cmb_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4, 5,], min=0, max=10)
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if freqmaps:
            outdir = "figs/freqmaps/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
    
            try:
                # 030 GHz IQU
                ctx.invoke(plot, input=f"BP_030_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_030_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_030_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3, 4, 5], min=0, max=75)
                ctx.invoke(plot, input=f"BP_030_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[6,], min=0, max=2, scale=1/3, title="$\sigma_{30}/3$",)
                ctx.invoke(plot, input=f"BP_030_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[7, 8], min=0, max=2)
                # 044 GHz IQU
                ctx.invoke(plot, input=f"BP_044_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_044_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_044_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,4,5,],min=0, max=75)
                ctx.invoke(plot, input=f"BP_044_IQU_n0512_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[6,7,8],min=0, max=2)
                # 070 GHz IQU
                ctx.invoke(plot, input=f"BP_070_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=3400,)
                ctx.invoke(plot, input=f"BP_070_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  fwhm=60.0, range=30,)
                ctx.invoke(plot, input=f"BP_070_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,4,5,], min=0, max=75, title="$A^{\mathrm{RMS}}_{30}/2$",scale=1/2)
                ctx.invoke(plot, input=f"BP_070_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[6, 7, 8], min=0, max=2,)

                # WMAP bands
                bandlabs = ['023-WMAP_K',  '030-WMAP_Ka', '040-WMAP_Q1', '040-WMAP_Q2',
                            '060-WMAP_V1', '060-WMAP_V2', '090-WMAP_W1', '090-WMAP_W2',
                            '090-WMAP_W3', '090-WMAP_W4']
                rms_lims = [150, 120, 160, 150, 
                            220, 200, 350, 370,
                            400, 380]
                for rms, lab in  zip(rms_lims, bandlabs):
                    ctx.invoke(plot, input=f"BP_{lab}_IQU_n0512_{procver}.fits",
                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[0,],  range=3400, scale=1e3)
                    ctx.invoke(plot, input=f"BP_{lab}_IQU_n0512_{procver}.fits",
                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[1, 2,],  fwhm=60.0, range=30,scale=1e3)
                    ctx.invoke(plot, input=f"BP_{lab}_IQU_n0512_{procver}.fits",
                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[3,4,5,],min=0, max=rms, scale=1e3)
                    ctx.invoke(plot, input=f"BP_{lab}_IQU_n0512_{procver}.fits",
                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[6,7,8],min=0, max=2, scale=1e3)


            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if synch:
            outdir = "figs/synchrotron/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            try:
                # Synch IQU
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0], min=10, mid=[100], max=200, scale=1e-6, unit="$\mathrm{K_{RJ}}$")
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,], )
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[6,], min=1, mid=[2], max=3, scale=1e-6,  unit="$\mathrm{K_{RJ}}$")
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[7, 8,], min=0, max=5)
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[9,], min=0, max=10)

                #P
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3], cmap="swamp", min=5, max=50, mid=[10, 25])


                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4, 5,], min=-3.4, max=-3.00, mid=[-3.2,], cmap="fusion" )
                ctx.invoke(plot, input=f"BP_synch_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[10,11], min=0, mid=[0.05,0.1], max=0.15, cmap="neutral_r")
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if ff:
            outdir = "figs/freefree/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            try:
                # freefree mean and rms
                ctx.invoke(plot, input=f"BP_freefree_I_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1,], )
                ctx.invoke(plot, input=f"BP_freefree_I_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[2,], min=0, mid=[25], max=50)
                #Dont plot te
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if ame:
            outdir = "figs/ame/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            try:
                # ame mean and rms
                ctx.invoke(plot, input=f"BP_ame_I_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,1], )
                ctx.invoke(plot, input=f"BP_ame_I_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1,], )
                ctx.invoke(plot, input=f"BP_ame_I_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[2,], min=0, max=80, mid=[40])
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if dust:
            outdir = "figs/dust/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
    
            try:
                # I Q U P IBETA QUBETA ITMEAN QUTMEAN   ISTDDEV QSTDDEV USTDDEV PSTDDEV    IBETASTDDEV QUBETASTDDEV ITSTDDEV QUTSTDDEV
                # dust IQU
                ctx.invoke(plot, input=f"BP_dust_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0, 1, 2, 4, 5, 6, 7,], )
                ctx.invoke(plot, input=f"BP_dust_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[8,], min=0, mid=[10], max=20)
                ctx.invoke(plot, input=f"BP_dust_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[9, 10], min=0, max=3)

                #P
                ctx.invoke(plot, input=f"BP_dust_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3], cmap="sunburst", min=5, max=50, mid=[10, 25])
                ctx.invoke(plot, input=f"BP_dust_IQU_n1024_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[11,], min=0, max=6)
            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if diff:
            outdir = "figs/freqmap_difference/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
                
            try:
                # Plot difference to npipe, dx12, BP10, and WMAP9
                ctx.invoke(plot, input=f"diffs/BP_030_diff_npipe_{procver}.fits", title="$\Delta A_{30}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_030_diff_npipe_{procver}.fits", title="$\Delta A_{30}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_030_diff_dx12_{procver}.fits",title="$\Delta A_{30}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_030_diff_dx12_{procver}.fits",title="$\Delta A_{30}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_030_diff_BP10_{procver}.fits",title="$\Delta A_{30}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_030_diff_BP10_{procver}.fits",title="$\Delta A_{30}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)

                ctx.invoke(plot, input=f"diffs/BP_044_diff_npipe_{procver}.fits", title="$\Delta A_{44}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_044_diff_npipe_{procver}.fits", title="$\Delta A_{44}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_044_diff_dx12_{procver}.fits",  title="$\Delta A_{44}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_044_diff_dx12_{procver}.fits",  title="$\Delta A_{44}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_044_diff_BP10_{procver}.fits",  title="$\Delta A_{44}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_044_diff_BP10_{procver}.fits",  title="$\Delta A_{44}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
            
                ctx.invoke(plot, input=f"diffs/BP_070_diff_npipe_{procver}.fits", title="$\Delta A_{70}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_070_diff_npipe_{procver}.fits", title="$\Delta A_{70}^{\mathrm{DR4}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_070_diff_dx12_{procver}.fits",  title="$\Delta A_{70}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_070_diff_dx12_{procver}.fits",  title="$\Delta A_{70}^{\mathrm{2018}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)
                ctx.invoke(plot, input=f"diffs/BP_070_diff_BP10_{procver}.fits",  title="$\Delta A_{70}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,],  range=10)
                ctx.invoke(plot, input=f"diffs/BP_070_diff_BP10_{procver}.fits",  title="$\Delta A_{70}^{\mathrm{BP10}}$", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,],  range=4)

                wbands = ["023-WMAP_K", "030-WMAP_Ka","040-WMAP_Q1","040-WMAP_Q2","060-WMAP_V1",
                          "060-WMAP_V2","090-WMAP_W1","090-WMAP_W2","090-WMAP_W3","090-WMAP_W4"]
                for wb in wbands:
                    lab = wb.split('_')[-1]
                    ctx.invoke(plot, input=f"diffs/BP_{wb}_diff_wmap9_{procver}.fits",title="$\Delta A_{" + lab + "}^{\mathit{WMAP9}}$", 
                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[0,],  range=10, scale=1e3)
                    ctx.invoke(plot, input=f"diffs/BP_{wb}_diff_wmap9_{procver}.fits",title="$\Delta A_{" + lab + "}^{\mathit{WMAP9}}$", 

                        size=size, outdir=outdir, colorbar=colorbar, auto=True,
                        sig=[1,2,],  range=4, scale=1e3)


            except Exception as e:
                print(e)
                click.secho("Continuing...", fg="yellow")

        if diffcmb:
            outdir = "figs/cmb_difference/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            mask = "/mn/stornext/u3/trygvels/compsep/cdata/like/BP_releases/masks/dx12_v3_common_mask_int_005a_1024_TQU.fits"
            mask2 =  "/mn/stornext/u3/hke/xsan/commander3/v2/chains_BP8_pol_resamp/mask_BP8_north_n8_v2.fits" #"/mn/stornext/u3/trygvels/compsep/cdata/like/paper_workdir/cmbdiffs/mask_dx12_and_BPproc.fits"
            mask3 = "/mn/stornext/u3/trygvels/compsep/cdata/like/paper_workdir/cmbdiffs/mask_bp8_full_IQU_n1024_v6.0_LFIPntSrc.fits"
            for i, method in enumerate(["Commander", "SEVEM", "NILC", "SMICA",]):
                try:
                    input = f"diffs/BP_cmb_diff_{method.lower()}_{procver}.fits"
                    ttl = "$\mathrm{"+method+"}$"
                    ctx.invoke(plot, input=input, size="s", unit='$\\mathrm{K}$', outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask3, remove_monopole=mask3, sig=[0,],  range=10, title=ttl, ltitle=" ", mask=mask3, mfill="gray",)
                    ctx.invoke(plot, input=input, size="s", unit='$\\mathrm{K}$', outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,], remove_monopole=mask2, range=4, title=ttl, ltitle=" ", mask=mask2, mfill="gray",)

                    ctx.invoke(plot, input=input, size="ml", unit='$\\mathrm{K}$', outdir=outdir, colorbar=colorbar, auto=True, remove_dipole=mask3, remove_monopole=mask3, sig=[0,],  range=10, title=ttl, ltitle=" ", mask=mask3, mfill="gray",)
                    ctx.invoke(plot, input=input, size="ml", unit='$\\mathrm{K}$', outdir=outdir, colorbar=colorbar, auto=True, sig=[1, 2,], remove_monopole=mask2, range=4, title=ttl, ltitle=" ", mask=mask2, mfill="gray",)
                except Exception as e:
                    print(e)
                    click.secho("Continuing...", fg="yellow")
        
        if goodness:
            import glob
            outdir = "figs/goodness/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
                           
            if goodness_temp:
                tbands = ["030_IQU", "044_IQU", "070_IQU", "023-WMAP_K",
                    "030-WMAP_Ka","040-WMAP_Q1","040-WMAP_Q2","060-WMAP_V1",
                    "060-WMAP_V2","090-WMAP_W1","090-WMAP_W2","090-WMAP_W3", 
                    "090-WMAP_W4","0.4-Haslam", "857",]
                
                for band in tbands:
                    try:
                        sig = [0,1] if not band in ["030","044","070"] else [0,3]
                        b = glob.glob(f'goodness/BP_res_{band}*fits')[0]
                        ctx.invoke(plot, input=b, size="msx", outdir=outdir, colorbar=colorbar, auto=True, sig=sig,)
                    except Exception as e:
                        print(e)
                        click.secho("Continuing...", fg="yellow")

            if goodness_pol:
                pbands = ["030_IQU", "044_IQU", "070_IQU", "023-WMAP_K",
                    "030-WMAP_Ka","040-WMAP_Q1","040-WMAP_Q2","060-WMAP_V1",
                    "060-WMAP_V2","090-WMAP_W1","090-WMAP_W2","090-WMAP_W3", 
                    "090-WMAP_W4","353"]
                mask_path='/mn/stornext/u3/trygvels/compsep/cdata/like/paper_workdir/synch/wmap_masks/'
                masks = ['wmap_processing_mask_Ka_r4_9yr_v5_TQU_chisq50.fits',  'wmap_processing_mask_Q_r4_9yr_v5_TQU_chisq50.fits', 'wmap_processing_mask_V_r4_9yr_v5_TQU_chisq50.fits',]
                m = 0
                for band in pbands:
                    try:
                        if (band in ["030_IQU","044_IQU","070_IQU"]) or ("WMAP" in band):
                             sig = [1,2,4,5]
                        else:
                             sig = [0,1,2,3]
                        b = glob.glob(f'goodness/BP_res_{band}*fits')[0]
                        if band in ["033-WMAP_Ka_P", "041-WMAP_Q_P", "061-WMAP_V_P",]:
                            ctx.invoke(plot, input=b, size="msx", outdir=outdir, colorbar=colorbar, auto=True, sig=sig, mask=mask_path+masks[m], mfill="gray")                
                            m+=1
                        else:
                            ctx.invoke(plot, input=b, size="msx", outdir=outdir, colorbar=colorbar, auto=True, sig=sig,)                
                    except Exception as e:
                        print(e)
                        click.secho("Continuing...", fg="yellow")

            if chisq:
                nsides = [16, 16, 16, 512, 512, 1024, 1024]
                fudge=1.02
                scale = fudge*2*(np.sum([(x/16)**2 for x in nsides])-3*(64/16)**2)
                ctx.invoke(plot, input=f"goodness/BP_chisq_n16_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[1,], scale=scale,)
                #ctx.invoke(plot, input=f"goodness/BP_chisq_n16_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[4,5], min=0.001, max=0.01, scale=scale)

                nsides = [512, 512, 1024, 512, 512, 512, 512, 512, 512, 1024]
                scale = (np.sum([(x/16)**2 for x in nsides])-3*(128/16)**2)
                ctx.invoke(plot, input=f"goodness/BP_chisq_n16_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[0,], scale=scale)
                #ctx.invoke(plot, input=f"goodness/BP_chisq_n16_{procver}.fits", size=size, outdir=outdir, colorbar=colorbar, auto=True, sig=[3,], min=0.001, max=0.01, scale=scale)

            
     
    if spec: 
        print("Plotting sky model SED spectrum")
        print("Reading data")
        import healpy as hp
        maskpath="/mn/stornext/u3/trygvels/compsep/cdata/like/sky-model/masks"
        fg_path="/mn/stornext/u3/trygvels/compsep/cdata/like/sky-model/fgs_60arcmin"
        
        a_cmb = None
        #BP_synch_IQU_n1024_BP8_noMedianFilter.fits
        a_s = hp.read_map(f"BP_synch_IQU_n1024_{procver}.fits", field=(0,1,2), dtype=None)
        b_s = hp.read_map(f"BP_synch_IQU_n1024_{procver}.fits", field=(4,5), dtype=None)
        
        a_ff = hp.read_map(f"BP_freefree_I_n1024_{procver}.fits", field=(0,), dtype=None)
        a_ff = hp.smoothing(a_ff, fwhm=arcmin2rad(np.sqrt(60.0**2-30**2)))
        t_e  = hp.read_map(f"BP_freefree_I_n1024_{procver}.fits", field=(1,), dtype=None)
        
        a_ame1 = hp.read_map(f"BP_ame_I_n1024_{procver}.fits", field=(0,), dtype=None)
        #a_ame1 = hp.smoothing(a_ame1, fwhm=arcmin2rad(np.sqrt(60.0**2-30**2)))
        nup    = hp.read_map(f"BP_ame_I_n1024_{procver}.fits", field=(1,), dtype=None)
        polfrac = 0.01
        a_ame2 = None
        
        a_d = hp.read_map(f"BP_dust_IQU_n1024_{procver}.fits", field=(0,1,2), dtype=None)
        a_d = hp.smoothing(a_d, fwhm=arcmin2rad(np.sqrt(60.0**2-10**2)))
        b_d = hp.read_map(f"BP_dust_IQU_n1024_{procver}.fits", field=(4,5,), dtype=None)
        t_d = hp.read_map(f"BP_dust_IQU_n1024_{procver}.fits", field=(6,7,), dtype=None)
        
        a_co10=f"{fg_path}/co10_npipe_60arcmin.fits"
        a_co21=f"{fg_path}/co21_npipe_60arcmin.fits"
        a_co32=f"{fg_path}/co32_npipe_60arcmin.fits"
        
        mask1=f"{maskpath}/mask_70GHz_t7.fits"
        mask2=f"{maskpath}/mask_70GHz_t100.fits"
                
        print("Data read, making plots, this may take a while")
        """
        for long in [False,True]:
            for pol in [True,False]:
                ctx.invoke(output_sky_model, pol=pol, long=long,
                           darkmode=False, png=False,
                           nside=64, a_cmb=a_cmb, a_s=a_s, b_s=b_s, a_ff=a_ff,
                           t_e=t_e, a_ame1=a_ame1, a_ame2=a_ame2, nup=nup, polfrac=polfrac, a_d=a_d, b_d=b_d,
                           t_d=t_d, a_co10=a_co10, a_co21=a_co21, a_co32=a_co32, mask1=mask1,
                           mask2=mask2,)
        """
        for long in [True, False]:
            for pol in [True, False]:
                ctx.invoke(output_sky_model, pol=pol, long=long,
                           darkmode=False, png=False,
                           nside=64, a_cmb=a_cmb, a_s=a_s, b_s=b_s, a_ff=a_ff,
                           t_e=t_e, a_ame1=a_ame1, a_ame2=a_ame2, nup=nup, polfrac=polfrac, a_d=a_d, b_d=b_d,
                           t_d=t_d, a_co10=a_co10, a_co21=a_co21, a_co32=a_co32, mask1=mask1,
                           mask2=mask2,)

        print("Renaming old files")
        outdir = "figs/sky-model/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        files = os.listdir(".")
        for f in files:
            if f.startswith("spectrum"):
                os.rename(f, f"{outdir}{f}")


@commands_plotting.command()
@click.argument("chainfile", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.option('-burnin', default=0, help='Min sample of dataset (burnin)')
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option("-lloc", default=0, help="Legend location",)
@click.option('-nbins', default=40, help='Bins for plotting')
@click.option('-sig', default="P", help='T or P')
@click.option('-idx', default=[0,], multiple=True, help='This is regions for synch')
@click.option('-prior', default=[], nargs=2, multiple=True, help='Specify mean and stddev')
@click.option('-min', default=None, type=click.FLOAT, help='Minimum valued on x')
@click.option('-max', default=None, type=click.FLOAT, help='Maximum valued on x')
def hist(chainfile, dataset, burnin, maxchain, lloc, nbins, sig, idx, prior, min, max):
    """
    Make histogram
    ex.
    c3pp hist synch/beta_pixreg_val -maxchain 2 -idx 1 -idx 3 -prior -3.3 0.1 -min -3.4 -max -2.8 -lloc 1 -nbins 40
    """
    # Check if you want to output a map
    import h5py
    import pandas as pd
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.colors as pcol
    from cycler import cycler
    colors=getattr(pcol.qualitative,"Plotly")

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    
    fontsize = 8
    WIDTH = 3.52

    # Get data
    dats = []
    for c in range(1, maxchain + 1):
        chainfile_ = chainfile.replace("c0001", "c" + str(c).zfill(4))
        min_=burnin if c>1 else 0
        with h5py.File(chainfile_, "r") as f:
            max_ = len(f.keys()) - 1
            print("{:-^48}".format(f" Samples {min_} to {max_} in {chainfile_} "))
            for sample in tqdm(range(min_, max_ + 1), ncols=80):
                # Identify dataset
                # HDF dataset path formatting
                s = str(sample).zfill(6)
                # Sets tag with type
                tag = f"{s}/{dataset}"
                #print(f"Reading c{str(c).zfill(4)} {tag}")
                # Check if map is available, if not, use alms.
                # If alms is already chosen, no problem
                try:
                    data = f[tag][()]
                    if len(data[0]) == 0:
                        print(f"WARNING! {tag} empty")
                    dats.append(data)
                except:
                    print(f"Found no dataset called {dataset}")
                    # Append sample to list
             
    df = pd.DataFrame.from_records(dats,columns=["T","P"])
    df2 = pd.DataFrame(df[sig].to_list())

    x = df2.to_numpy()

    fig = plt.figure(figsize=(WIDTH,0.75*WIDTH))

    if "synch" in dataset:
        labels=['High lat.', 'Spur', 'Center', 'Plane', ]
        bins = np.linspace(min,max,nbins)
        for i, idx in enumerate(idx):
            if i==0:
                n1, _, _ = plt.hist(x[:,idx],bins=bins, histtype='step', color=f"k", density=True, stacked=True, label=r"$P(\beta^{\mathrm{"+labels[idx]+r"}}_{\mathrm{s}}|\,d)$")
                ymax=np.max(n1)
            else:
                nn, _, _ = plt.hist(x[:,idx],bins=bins, histtype='step', color=f"k", linestyle="--", density=True, stacked=True, label=r"$P(\beta^{\mathrm{"+labels[idx]+r"}}_{\mathrm{s}}|\,d)$")
                if np.max(nn)>ymax:
                    ymax=np.max(nn)
        """
        #t3 = np.loadtxt("sampletrace_t3.csv",delimiter=",",) #"regdatat3.dat", delimiter=",")
        t3 = np.loadtxt("regdatat3.dat", delimiter=",")
        plt.hist(t3[:,4],  bins=bins, histtype='step', density=True, stacked=True, linestyle="--", color="#ef553b", label=r"$P_{-2.8}(\beta^{\mathrm{Spur}}_{\mathrm{s}}|\,d, \omega_{\mathrm{TOD}})$")
        bp8r = np.loadtxt("sampletrace_P_synch-beta_pixreg_val.csv", delimiter=",", skiprows=1)
        plt.hist(bp8r[:,2],bins=bins, histtype='step', density=True, stacked=True, linestyle="--", color="#636efa", label=r"$P_{-3.1}(\beta^{\mathrm{Spur}}_{\mathrm{s}}|\,d, \omega_{\mathrm{TOD}})$")
        #n1, bins, _ = plt.hist(x[:,4],bins=20, histtype='step', color="#636efa", label=r"$P(\beta^{\mathrm{Spur}}_{\mathrm{s}}|\,d)$")
        #xmin, xmax = (-3.25,-2.75)
        """
        if prior!=[]:
            for i, pri in enumerate(prior):
                import scipy.stats as stats
                N = len(x[:,1])
                xs = np.linspace(min,max,N)
                dx = bins[1]-bins[0]
                norm = sum(n1)*dx
                #Pprior = stats.norm.pdf(x, -2.8, 0.1)#*norm
                #plt.plot(x, Pprior*norm, color="#ef553b", linestyle=":", label=r"$P(\beta_{\mathrm{s}})=\mathcal{N}(-2.8,0.1)$")
                Pprior = stats.norm.pdf(xs, float(pri[0]), float(pri[1]))*norm
                plt.plot(xs, Pprior, color=f"k", linestyle=":", label=r"$P(\beta_{\mathrm{s}})=\mathcal{N}("+f"{pri[0]}, {pri[1]}"+")$")

        plt.xlabel(r"Synchrotron index, $\beta_{\mathrm{s}}$", fontsize=fontsize)
        
    elif "dust" in dataset:
        bins = np.linspace(min,max,nbins)
        n, bins, _ = plt.hist(x,bins=bins, histtype='step', color="black", label=r"$P(\beta_{\mathrm{d}}|\,d)$")
        ymax = np.max(n)
        N = len(x)


        if prior!=[]:
            for pri in prior:
                import scipy.stats as stats
                N = len(x)
                xs = np.linspace(min,max,N)
                dx = bins[1]-bins[0]
                norm = sum(n)*dx
                Pprior = stats.norm.pdf(xs, float(pri[0]), float(pri[1]))*norm
                plt.plot(xs, Pprior, color="black", linestyle=":", label=r"$P(\beta_{\mathrm{d}})=\mathcal{N}("+f"{pri[0]}, {pri[1]}"+")$")
        plt.xlabel(r"Thermal dust index, $\beta_{\mathrm{d}}$", fontsize=fontsize)

    else: 
        n1, bins, _ = plt.hist(x[:,1],bins=nbins, histtype='step', density=True, stacked=True)
        ymax=np.max(n1)
        for i in range(x.shape[1]):
            nn, _, _, = plt.hist(x[:,i],bins=bins, histtype='step', density=True, stacked=True)
            if np.max(nn)>ymax:
                ymax=np.max(nn)

        if prior!=[]:
            for pri in prior:
                import scipy.stats as stats
                N = len(x[:,1])
                xs = np.linspace(min,max,N)
                dx = bins[1]-bins[0]
                norm = sum(n1)*dx
                #Pprior = stats.norm.pdf(x, -2.8, 0.1)#*norm
                #plt.plot(x, Pprior*norm, color="#ef553b", linestyle=":", label=r"$P(\beta_{\mathrm{s}})=\mathcal{N}(-2.8,0.1)$")
                Pprior = stats.norm.pdf(xs, float(pri[0]), float(pri[1]))*norm
                ymax=np.max(Pprior)*2.5
                plt.plot(xs, Pprior, color=f"C{i}", linestyle=":", label=r"$\mathcal{N}("+f"{pri[0]}, {pri[1]}"+")$")


    plt.xlim(min,max)

    ax=plt.gca()
    #ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylim(0,ymax*1.1)
    #plt.ylabel(r"Normalized number of samples", fontsize=fontsize)
    plt.legend(frameon=False,fontsize=fontsize, loc=lloc)
    plt.yticks(rotation=90, va="center", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(dataset.replace("/","-")+"_histogram.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)


@commands_plotting.command()
@click.argument('filename', type=click.STRING)
@click.option('-min', default=0, help='Min sample of dataset (burnin)')
@click.option('-max', default=1000, help='Max sample to inclue')
@click.option('-nbins', default=1, help='Bins')
@click.option("-cmap", default="Plotly", help='Sets colorcycler using plotly', type=click.STRING)
def traceplot(filename, max, min, nbins, cmap):
    """
    Traceplot of samples from .dat. 
    Accepts min to max with optional bins.
    Useful to plot sample progression of spectral indexes.
    """
    header = ['Prior', 'High lat.', 'Spur',
         'Center', 'Fan region', 'Anti-center',
         'Gum nebula']
    cols = [4,5,9,10,11,12,13]
    import pandas as pd
    df = pd.read_csv(filename, sep=r"\s+", usecols=cols, skiprows=range(min), nrows=max)
    df.columns = header
    x = 'MCMC Sample'
    
    traceplotter(df, header, x, nbins, outname=filename.replace(".dat","_traceplot.pdf"), min_=min, cmap=cmap)



@commands_plotting.command()
@click.argument("chainfile", type=click.STRING)
@click.argument("dataset", type=click.STRING)
@click.option('-burnin', default=0, help='Min sample of dataset (burnin)')
@click.option("-maxchain", default=1, help="max number of chains c0005 [ex. 5]",)
@click.option('-plot', is_flag=True, default=False, help= 'Plots trace')
@click.option('-freeze', is_flag=True, help= 'Freeze top regions')
@click.option('-nbins', default=1, help='Bins for plotting')
@click.option("-f", "priorsamp",  multiple=True, help="These are sampled around prior and will be marked",)
@click.option('-scale', default=0.023, help='scale factor for labels')
@click.option("-cmap", default="Plotly", help='sets colorcycler using Plotly', type=click.STRING)
def pixreg2trace(chainfile, dataset, burnin, maxchain, plot, freeze, nbins, priorsamp, scale,cmap):
    """
    Outputs the values of the pixel regions for each sample to a dat file.
    ex. c3pp pixreg2trace chain_c0001.h5 synch/beta_pixreg_val -burnin 30 -maxchain 4 
    """
    
    # Check if you want to output a map
    import h5py
    import healpy as hp
    import pandas as pd
    from tqdm import tqdm
    dats = []
    for c in range(1, maxchain + 1):
        chainfile_ = chainfile.replace("c0001", "c" + str(c).zfill(4))
        min_=burnin if c>1 else 0
        with h5py.File(chainfile_, "r") as f:
            max_ = len(f.keys()) - 1
            print("{:-^48}".format(f" Samples {min_} to {max_} in {chainfile_} "))
            for sample in tqdm(range(min_, max_ + 1), ncols=80):
                # Identify dataset
                # HDF dataset path formatting
                s = str(sample).zfill(6)
                # Sets tag with type
                tag = f"{s}/{dataset}"
                #print(f"Reading c{str(c).zfill(4)} {tag}")
                # Check if map is available, if not, use alms.
                # If alms is already chosen, no problem
                try:
                    data = f[tag][()]
                    if len(data[0]) == 0:
                        print(f"WARNING! {tag} empty")
                    dats.append(data)
                except:
                    print(f"Found no dataset called {dataset}")
                    # Append sample to list
             
    
 
    if "bp_delta" in dataset or "bandpass" in dataset:
        label = dataset.replace("/","-")
        outname = f"sampletrace_{label}"

        df = pd.DataFrame.from_records(dats,)
        nregs = len(df[0][0])
        if "30" in dataset:
            header = ["0","27M", "27S", "28M", "28S"]
        elif "44" in dataset:
            header = ["0","24M","24S","25M","25S","26M","26S"]
        elif "70" in dataset:
            header = ["0","18M","18S","19M","19S","20M","20S","21M","21S","22M","22S","23M","23S"]
        else:
            header = [str(i) for i in range(nregs)]

        df2 = pd.DataFrame(df[0].to_list(), columns=header)
        df2 = df2.drop(columns=["0",])
        df2 = df2*1e3 # Scale to MHz
        header = header[1:]
        traceplotter(df2, header, "Gibbs Sample", nbins, f"{outname}.pdf", min_=burnin,ylabel=r"$\delta$ bandpass [MHz]", priorsamp=priorsamp, scale=scale, cmap=cmap, figsize=(12,4), labscale=0.9)
    else:
        sigs = ["T","P"]
        df = pd.DataFrame.from_records(dats, columns=sigs)
        nregs = len(df["P"][0])
        
        if nregs == 9:
            header = ['Top left', 'Top right', 'Bot. left', 'Bot. right', 'Spur',
                      'Center', 'Fan region', 'Anti-center',
                      'Gum nebula']
        elif nregs == 6:
            header = ['High lat.', 'Spur',
                      'Center', 'Fan', 'Anti-center',
                      'Gum nebula']
        elif nregs == 4:
            header = ['High lat.', 'Spur',
                      'Gal. center', 'Gal. plane']
        elif nregs == 1:
            header = ['Fullsky',]
        else:
            header = [str(i) for i in range(nregs)]
        
        
        for sig in sigs:
            if sig =="T":
                continue
            label = dataset.replace("/","-")
            outname = f"sampletrace_{sig}_{label}"

            df2 = pd.DataFrame(df[sig].to_list(), columns=header)
            df2.to_csv(f'{outname}.csv')
            print(df2.shape)
        
            if plot:
                xlabel = 'Gibbs Sample'
                if freeze:
                    combined_hilat = 'High lat.'
                    df2 = df2.drop(columns=['Top left', 'Top right', 'Bot. left',])
                    df2 = df2.rename(columns={'Bot. right':combined_hilat})
                    header_ = [combined_hilat] + header[4:]
                else:
                    header_ = header.copy()
        
                traceplotter(df2, header_, xlabel, nbins, f"{outname}.pdf", min_=burnin, ylabel='Region spectral index', priorsamp=priorsamp, scale=scale, cmap=cmap)

def traceplotter(df, header, xlabel, nbins, outname, min_,ylabel, cmap="Plotly", priorsamp=None, scale=0.023, figsize=(16,8), labscale=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.colors as pcol
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.2})
    sns.set_style("whitegrid")
    plt.rcParams["mathtext.fontset"] = "stix"
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'font.size': 20, 
        'axes.linewidth': 1.5,
        'text.usetex': True,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': 'Times',
    }
    sns.set_style(custom_style)

    #df.columns = y
    N = df.values.shape[0]
    nregs = len(header)
    """
    df['Mean'] = df.mean(axis=1)
    header.append('Mean')
    """
    df[xlabel] = range(N)
    f, ax = plt.subplots(figsize=figsize)
    
    #cmap = plt.cm.get_cmap('tab20')# len(y))
    import matplotlib as mpl
    # Swap colors around
    colors=getattr(pcol.qualitative,cmap)    
    if cmap=="Plotly":
        colors.insert(0,colors.pop(-1))
        colors.insert(3,colors.pop(2))
        colors = colors + colors

    cmap = mpl.colors.ListedColormap(colors)
    #cmap = plt.cm.get_cmap('tab10')# len(y))
    df = df[min_:]
    means = df[min_:].mean()
    stds = df[min_:].std()
    # Reduce points
    if nbins>1:
        df = df.groupby(np.arange(len(df))//nbins).mean()
    
    #longestlab = max([len(x) for x in header])
    # scale = 0.075 for 9 regions worked well.
    positions = legend_positions(df, header, scaling=scale)
    c = 0
    for i, (column, position) in enumerate(positions.items()):
        linestyle = ':' if str(i) in priorsamp else '-'
        linewidth = 2
        fontweight = 'normal'
        """
        if column == "Mean":
            color="#a6a6a6" #"grey"
            linewidth = 4
            fontweight='bold'
        else:
            color = cmap(c)#float(i-1)/len(positions))
            c += 1
        """
        color = cmap(i)
    
        # Plot each line separatly so we can be explicit about color
        ax = df.plot(x=xlabel, y=column, legend=False, ax=ax, color=cmap(i), linestyle=linestyle, linewidth=linewidth,)

        label1 = rf'{column}'
        label2 = rf'{means[i]:.2f}$\pm${stds[i]:.2f}'
        # Add the text to the right
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*0.01,
            position, label1, fontsize=15,
            color=color, fontweight=fontweight
        )
        plt.text(
            df[xlabel][df[column].last_valid_index()]+N*0.13*labscale,
            position, label2, fontsize=15,
            color=color, fontweight='normal'
        )

    #if min_:
    #    plt.xticks(list(plt.xticks()[0]) + [min_])

    ax.set_ylabel(ylabel)
    #plt.yticks(rotation=90)
    plt.gca().set_xlim(right=N)
    #ax.axes.xaxis.grid()
    #ax.axes.yaxis.grid()
    # Add percent signs
    #ax.set_yticklabels(['{:3.0f}%'.format(x) for x in ax.get_yticks()])
    sns.despine(top=True, right=True, left=True, bottom=True)
    plt.subplots_adjust(wspace=0, hspace=0.01, right=0.81)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()


@commands_plotting.command()
@click.argument('dir1', type=click.STRING)
@click.argument('type1', type=click.Choice(['ml', 'mean']))
@click.argument('dir2', type=click.STRING)
@click.argument('type2', type=click.Choice(['ml', 'mean']))
@click.pass_context
def make_diff_plots(ctx, dir1, dir2, type1, type2):
    """
    Produces difference maps between output directories.
    """

    comps = ['030', '044', '070', 'ame', 'cmb', 'freefree', 'synch']

    filenames = {dir1:'', dir2:''}

    import glob
    import healpy as hp


    for dirtype, dirloc in zip([type1, type2],[dir1, dir2]):
        if dirtype == 'ml':
            #determine largest sample number
            cmbs = glob.glob(os.path.join(dirloc, 'cmb_c0001_k??????.fits'))
            indexes = [int(cmb[-11:-5]) for cmb in cmbs]
            intdexes = [int(index) for index in indexes]
            index = max(intdexes)
            
            filenames[dirloc] = '_c0001_k' + str(index).zfill(6) + '.fits'

    for comp in comps:
        mapn = {dir1:'', dir2:''}

        for dirtype, dirloc in zip([type1, type2],[dir1, dir2]): 
            #print(filenames[dirloc])
            if len(filenames[dirloc]) == 0:
                mapn[dirloc] = glob.glob(os.path.join(dirloc, 'BP_' + comp + '_I*.fits'))[0]
            else:
                if comp in ['ame', 'cmb', 'synch', 'dust']:
                    mapn[dirloc] = comp + filenames[dirloc]
                elif comp in ['freefree']:
                    mapn[dirloc] = 'ff' + filenames[dirloc]
                else:
                    mapn[dirloc] = 'tod_' + comp + '_map' + filenames[dirloc]
      
        map1 = hp.read_map(os.path.join(dir1, mapn[dir1]))
        map2 = hp.read_map(os.path.join(dir2, mapn[dir2]))

        diff_map = map1 - map2 
  
        from cosmoglobe.release.plotter import trygveplot
 
        size = 'm'
        if comp in ['030', '044', '070']:
            size = 's'

        trygveplot(diff_map, outname=comp + '_diff' + '.pdf', auto=True, rng='auto', colorbar=True, size=size,)

@commands_plotting.command()
@click.option("-pol", is_flag=True, help="",)
@click.option("-long", is_flag=True, help="",)
@click.option("-darkmode", is_flag=True, help="",)
@click.option("-png", is_flag=True, help="",)
@click.option("-nside", type=click.INT, help="",)
@click.option("-a_cmb", help="",)
@click.option("-a_s",  help="",)
@click.option("-b_s",  help="",)
@click.option("-a_ff", help="",)
@click.option("-t_e",  help="",)
@click.option("-a_ame1",help="",)
@click.option("-a_ame2", help="",)
@click.option("-nup",  help="",)
@click.option("-polfrac",  help="",)
@click.option("-a_d", help="",)
@click.option("-b_d", help="",)
@click.option("-t_d", help="",)
@click.option("-a_co10", help="",)
@click.option("-a_co21", help="",)
@click.option("-a_co32", help="",)
@click.option("-mask1",  help="",)
@click.option("-mask2",  help="",)
def output_sky_model(pol, long, darkmode, png, nside, a_cmb, a_s, b_s, a_ff, t_e, a_ame1, a_ame2, nup, polfrac, a_d, b_d, t_d, a_co10, a_co21, a_co32, mask1, mask2):
    """
    Outputs spectrum plots.
    c3pp output-sky-model -a_s synch_c0001_k000100.fits -b_s synch_beta_c0001_k000100.fits -a_d dust_init_kja_n1024.fits -b_d dust_beta_init_kja_n1024.fits -t_d dust_T_init_kja_n1024.fits -a_ame1 ame_c0001_k000100.fits -nup ame_nu_p_c0001_k000100.fits -a_ff ff_c0001_k000100.fits -t_e ff_Te_c0001_k000100.fits -mask1 mask_70GHz_t70.fits -mask2 mask_70GHz_t7.fits -nside 16
    """
    from cosmoglobe.release.spectrum import Spectrum
    """
    if not a_cmb:
        a_cmb = 0.67 if pol else 45
    if not a_s:
        a_s = 12 if pol else 76
    if not b_s:
        b_s = -3.1
    if not a_ff:
        a_ff = 30.
    if not t_e:
        t_e = 7000.
    if not a_ame1:
        a_ame1 = 5 if pol else 50
    if not a_ame2:
        a_ame2 = 50.
    if not nup:
        nup = 24
    if not polfrac:
        polfrac = 1
    if not a_d:
        a_d = 8 if pol else 163
    if not b_d:
        b_d = 1.6
    if not t_d:
        t_d = 18.5
    if not a_co10:
        a_co10=50
    if not a_co21:
        a_co21=25
    if not a_co32:
        a_co32=10
    """
    if pol:
        # 15, 120, 40, (0,4, 12), (1.2,50)
        p = 0.6 if long else 15
        sd = 2.5 if long else 70
        s=20 if long else 30
        foregrounds = {
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s,],
                             "position": s,
                             "color"   : "C2",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Thermal Dust": {"function": "tdust", 
                             "params": [a_d, b_d, t_d, 353],
                             "position": 250,
                             "color":    "C1",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         }, 
            "Sum fg."      : {"function": "sum", 
                             "params"  : [],
                             "position": 70,
                             "color"   : "C10",
                             "sum"     : False,
                             "linestyle": "--",
                             "gradient": False,
                          },
            r"BB $r=10^{-2}$"   :  {"function": "rspectrum", 
                             "params"  : [0.01, "BB",],
                             "position": p,
                             "color"   : "C10",
                             "sum"     : False,
                             "linestyle": "dotted",
                             "gradient": True,
                         },
            r"BB $r=10^{-4}$"   :  {"function": "rspectrum", 
                             "params"  : [1e-4, "BB",],
                             "position": p,
                             "color"   : "C10",
                             "sum"     : False,
                             "linestyle": "dotted",
                             "gradient": True,
                         },
            "CMB EE":       {"function": "rspectrum", 
                             "params"  : [1, "EE"],
                             "position": p,
                             "color"   : "C5",
                             "sum"     : False,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Spinning Dust" : {"function": "sdust", 
                               "params"  : [a_ame1, nup, polfrac],
                             "position": sd,
                             "color"   : "C4",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": True,
                         },

            }
    else:
        #120, 12, 40, (2,57), 20, 70
        p = 3 if long else 65
        td = 10 if long else 17
        s = 2 if long else 120
        f=17 if long else 22
        sfg=25 if long else 70
        foregrounds = {
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s, 0.4e9],
                             "position": s,
                             "color"   : "C2",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Thermal Dust": {"function": "tdust", 
                             "params": [a_d, b_d, t_d, 545],
                             "position": td,
                             "color":    "C1",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         }, 
            "Free-Free"  : {"function": "ff", 
                             "params"  : [a_ff, t_e],
                             "position": f,
                             "color"   : "C0",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            "Spinning Dust" : {"function": "sdust", 
                            "params"  : [a_ame1, nup, 1.],
                             "position": p,
                             "color"   : "C4",
                             "sum"     : True,
                             "linestyle": "solid",
                             "gradient": False,
                         },
            r"CO$_{1\rightarrow 0}$": {"function": "line", 
                                       "params"  : [a_co10, 115, 11.06],
                                       "position": p,
                                       "color"   : "C9",
                                       "sum"     : True,
                                       "linestyle": "solid",
                                       "gradient": False,
                         },
            r"CO$_{2\rightarrow 1}$": {"function": "line", 
                                       "params"  : [a_co21, 230., 14.01],
                                       "position": p,
                                       "color"   : "C9",
                                       "sum"     : True,
                                       "linestyle": "solid",
                                       "gradient": False,
                         },
            r"CO$_{3\rightarrow 2}$":      {"function": "line", 
                                            "params"  : [a_co32, 345., 12.24],
                                            "position": p,
                                            "color"   : "C9",
                                            "sum"     : True,
                                            "linestyle": "solid",
                                            "gradient": False,
                         },
            "Sum fg."      : {"function": "sum", 
                             "params"  : [],
                             "position": sfg,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "--",
                             "gradient": False,
                          },
            "CMB":          {"function": "rspectrum", 
                             "params"  : [1., "TT"],
                             "position": 70,
                             "color"   : "C5",
                             "sum"     : False,
                             "linestyle": "solid",
                             "gradient": False,
                         },

            }

    Spectrum(pol, long, darkmode, png, foregrounds, [mask1,mask2], nside)



