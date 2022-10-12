import click
from rich import print
import astropy.units as u
import matplotlib.pyplot as plt
import os
import re

from cosmoglobe.plot import plot as cplot
from cosmoglobe.plot import trace as ctrace
from cosmoglobe.plot import gnom as cgnom


from .plottools import *


@click.group()
def commands_plotting():
    pass

# fmt: off
@commands_plotting.command()
@click.argument("input", nargs=-1)
@click.option("-sig", default = 0, help="Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)",)
@click.option("-comp", default = None, type= click.STRING,help="Component name for autosetting plottingparameters",)
@click.option("-freq", default = None, type= click.FLOAT,help="Frequency in GHz needed for scaling maps when using a model object input",)
@click.option("-ticks", default = None, help="Min and max value for data. If None, uses 97.5th percentile.",)
@click.option("-min", default = None, type= click.FLOAT,help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", default = None, type= click.FLOAT,help="Max value of colorbar, overrides autodetector.",)
@click.option("-range", "rng",  default = None,  type= click.FLOAT, help="Color bar range")
@click.option("-nocbar",is_flag=True,help='Adds colorbar ("cb" in filename)',)
@click.option("-unit", default = None, type= click.STRING,help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-fwhm", default = 0.0,type= click.FLOAT,help="FWHM of smoothing, in arcmin.",)
@click.option("-nside", default = None, type= click.INT,help="nside for optional ud_grade.",)
@click.option("-sample", default = -1, type= click.INT,help="Select sample when inputing .h5 file",)
@click.option("-mask", default = None, type= click.STRING,help="Masks input with specified maskfile.",)
@click.option("-maskfill", default = None, type= click.STRING,help='Color to fill masked area. for example "gray". Transparent by default.',)
@click.option("-cmap", default = None, type= click.STRING,help="Chose colormap (ex. sunburst, planck, etc). Available are matplotlib and cmasher. Also qualitative plotly [ex. q-Plotly-4 (q for qualitative 4 for max color)]",)
@click.option("-norm", default = None, help='"Linear" is normal, "log" Normalizesusingsemi-logscale',)
@click.option("-remove_dip",is_flag=True,help="Fits a dipole to the map and removes it.",)
@click.option("-remove_mono",is_flag=True,help="Fits a monopole to the map and removes it.",)
@click.option("-title", default = None, type= click.STRING,help="Sets the full figure title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-rlabel", default = None, type= click.STRING,help="Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-llabel", default = None, type= click.STRING,help="Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-width", default = None, help="Size in inches. Default is fraction=0.5",)
@click.option("-fraction", default = 0.5,type= click.FLOAT,help="Fraction of latex page width",)
@click.option("-xsize", default = 1000, type= click.INT,help="figuresize in px",)
@click.option("-darkmode",is_flag=True,help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',)
#@click.option("-interactive",is_flag=True,help="Uses healpys interactive mode, zooming is possible etc.",)
@click.option("-rot", default = None, help="Describe the rotation to apply. In the form (lon, lat, psi) (unit: degrees) : the point at longitude *lon* and latitude *lat* will be at the center. An additional rotation of angle *psi* around this direction is applied.",)
@click.option("-coord", default = None, help="Either one of 'G', 'E' or 'C' to describe the coordinate system of the map,",)
@click.option("-nest",is_flag=True,help="If True, ordering scheme is NESTED.",)
@click.option("-flip", default = "astro" ,type= click.STRING,help="Defines the convention of projection : 'astro' or 'geo'",)
@click.option("-graticule",is_flag=True,help="Adds graticule",)
@click.option("-graticule_labels",is_flag=True,help="longitude and latitude labels",)
@click.option("-return_only_data",is_flag=True,help="Returns figure",)
@click.option("-projection_type", default = "mollweide" ,type= click.STRING,help='{"aitoff", "hammer", "lambert", "mollweide", "cart", "3d", "polar"} type of the plot',)
@click.option("-cb_orientation", default = "horizontal" ,type= click.STRING,help='{"horizontal", "vertical"} color bar orientation',)
@click.option("-xlabel", default = None, type= click.STRING,help="set x axis label",)
@click.option("-ylabel", default = None, type= click.STRING,help="set y axis label",)
@click.option("-longitude_grid_spacing", default = 60, type= click.FLOAT,help="set x axis grid spacing",)
@click.option("-latitude_grid_spacing", default = 30, type= click.FLOAT,help="set y axis grid spacing",)
@click.option("-override_plot_properties", default = None, help='Override the following plot proporties: "cbar_shrink", "cbar_pad", "cbar_label_pad", "figure_width": width, "figure_size_ratio": ratio.',)
@click.option("-xtick_label_color", default = "black" ,type= click.STRING,help="color of longitude ticks",)
@click.option("-ytick_label_color", default = "black" ,type= click.STRING,help="color of latitude ticks",)
@click.option("-graticule_color", default = None, type= click.STRING,help="color of latitude ticks",)
@click.option("-fontsize", default = None, help="Dictionary with fontsizes: 'xlabel', 'ylabel', 'title', 'xtick_label', 'ytick_label', 'cbar_label', 'cbar_tick_label'.",)
@click.option("-phi_convention", default = "counterclockwise" ,type= click.STRING,help='convention on x-axis (phi), "counterclockwise" (default), "clockwise", "symmetrical" (phi as it is truly given) if "flip" is "geo", "phi_convention" should be set to "clockwise".',)
@click.option("-custom_xtick_labels", default = None, help="override x-axis tick labels",)
@click.option("-custom_ytick_labels", default = None, help="override y-axis tick labels",)
@click.option("-white_background",is_flag=True,help="Sets the background to be white. (Transparent by default [recommended])",)
@click.option("-png",is_flag=True,help="Saves output as .png ().pdf by default)",)
@click.option("-outdir", default = None, type= click.Path(exists=True),help="Output directory for plot",)
@click.option("-outname", default = None, type= click.STRING,help="Output filename, overwrites autonaming",)
@click.option("-show",is_flag=True,help="Displays the figure in addition to saving",)
@click.option("-gif",is_flag=True,help="Convert inputs to gif",)
@click.option("-fps", default = 3, type= click.INT,help="Gif speed, 300 by default.",)
@click.option("-dpi", default = 300, type= click.INT,help="DPI of output file. 300 by default.",)
# fmt: on
def plot(input, **kwargs):
    """
    Invokes the plot function from the command line
    input: filename of fits or h5 file, optional plotting paramameters
    """
    kwargs["cbar"] = not kwargs.pop("nocbar")
    kwargs["freq"] = None if kwargs["freq"] is None else kwargs["freq"] * u.GHz
    kwargs["fwhm"] = None if kwargs["fwhm"] is None else kwargs["fwhm"] * u.arcmin

    # Pop click-specific arguments
    gif = kwargs.pop("gif")
    show = kwargs.pop("show")
    dpi = kwargs.pop("dpi")
    fps = kwargs.pop("fps")
    png = kwargs.pop("png")
    white_background = kwargs.pop("white_background")
    outdir = kwargs.pop("outdir")
    outname = kwargs.pop("outname")

    imgs = []  # Container for images used with gif
    for i, filename in enumerate(input):
        # Super hacky gif fix because "hold" is not implemented in newvisufunc
        if gif and i > 0:
            kwargs["return_only_data"] = True

        # Actually plot the stuff
        img, params = cplot(filename, **kwargs)

        # Super hacky gif fix because "hold" is not implemented in newvisufunc
        if gif:
            if i > 0:
                longitude, latitude, grid_map = img
                img = plt.pcolormesh(
                    longitude,
                    latitude,
                    grid_map,
                    vmin=params["ticks"][0],
                    vmax=params["ticks"][-1],
                    rasterized=True,
                    cmap=load_cmap(params["cmap"]),
                    shading="auto",
                )

            """
            # Show title for gifs
            sample = re.search('_k([0-9]*)', filename)
            if sample:
                sample = sample.group(1).lstrip("0")
            else:
                sample = filename
            """
        if show:
            plt.show()

        """
        Filename formatting and  outputting
        """
        *path, fn = os.path.split(filename)
        fn, ftype = os.path.splitext(fn)

        comp = kwargs["comp"]
        freq = kwargs["freq"]
        fwhm = kwargs["fwhm"]

        if ftype == ".h5":
            fn += "_" + comp

        fn = fn.replace("_IQU_", "_").replace("_I_", "_")
        if freq is not None:
            fn = fn.replace(comp, f"{comp}-{int(freq.value)}GHz")
        if outdir:
            path = outdir

        tags = []
        tags.append(f"{str(int(fwhm.value))}arcmin") if float(fwhm.value) > 0 else None
        tags.append("cb") if kwargs["cbar"] else None
        tags.append("masked") if kwargs["mask"] else None
        tags.append("nodip") if kwargs["remove_dip"] else None
        tags.append("dark") if kwargs["darkmode"] else None
        if params["cmap"] is None:
            params["cmap"] = "planck"
        tags.append(f'c-{params["cmap"]}')

        nside_tag = "_n" + str(int(params["nside"]))
        if nside_tag in fn:
            fn = fn.replace(nside_tag, "")

        if isinstance(kwargs["sig"], int):
            sig = ["I", "Q", "U"][kwargs["sig"]]
        fn = fn + f'_{sig}_w{str(int(params["width"]))}' + nside_tag

        for i in tags:
            fn += f"_{i}"

        filetype = "png" if png else "pdf"
        if outname:
            filetype = "png" if outname.endswith(".png") else "pdf"

        fn += f".{filetype}"
        if outname:
            fn = outname

        if outdir:
            path = [outdir]

        if gif:
            # Append images for animation
            imgs.append([img])
        else:
            tp = False if white_background else True
            filetype = "png" if png else "pdf"
            path = "." if path[0] == "" else path[0]
            print(f"[bold green]Outputting {fn}[/bold green] to {path}")
            plt.savefig(
                f"{path}/{fn}",
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=tp,
                format=filetype,
                dpi=dpi,
            )

    if gif:
        import matplotlib.animation as animation

        ani = animation.ArtistAnimation(plt.gcf(), imgs, blit=True)
        fn = fn.replace(filetype, "gif")
        print(f"[bold green]Outputting {fn}[/bold green]")
        ani.save(fn, dpi=dpi, writer=animation.PillowWriter(fps=fps))

# fmt: off
@commands_plotting.command()
@click.argument("input", type=click.STRING)
@click.option("-lon", default=0, type=click.INT)
@click.option("-lat", default=0, type=click.INT)
@click.option( "-comp", default=None, type=click.STRING, help="Component name for autosetting plottingparameters",)
@click.option( "-sig", default=0, help="Which sky signal to plot",)
@click.option( "-freq", default=None, type=click.FLOAT, help="Frequency in GHz needed for scaling maps when using a model object input",)
@click.option( "-nside", default=None, type=click.INT, help="nside for optional ud_grade.",)
@click.option("-size", default=20, type=click.INT)
@click.option( "-sample", default=-1, type=click.INT, help="Select sample when inputing .h5 file",)
@click.option( "-min", "vmin", default=None, type=click.FLOAT, help="Min value of colorbar, overrides autodetector.",)
@click.option( "-max", "vmax", default=None, type=click.FLOAT, help="Max value of colorbar, overrides autodetector.",)
@click.option( "-ticks", default=None, help="Min and max value for data. If None, uses 97.5th percentile.",)
@click.option("-range", "rng", default=None, type=click.FLOAT, help="Color bar range")
@click.option( "-llabel", default=None, type=click.STRING, help="Set title, has LaTeX functionality. Ex. $\mu$",)
@click.option( "-rlabel", default=None, type=click.STRING, help="Set title, has LaTeX functionality. Ex. $\mu$",)
@click.option( "-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option( "-cmap", default=None, help="Choose different color map (string), such as Jet or planck",)
@click.option( "-graticule", is_flag=True, help="Add graticule",)
@click.option( "-norm", default=None, help='"Linear" is normal, "log" Normalizesusingsemi-logscale',)
@click.option( "-nocbar", is_flag=True, help="remove colorbar",)
@click.option( "-cbar_pad", default=0.04, type=click.FLOAT, help="Colorbar space",)
@click.option( "-cbar_shrink", default=0.7, type=click.FLOAT, help="Colorbar size",)
@click.option( "-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing, in arcmin.",)
@click.option( "-remove_dip", default=None, type=click.STRING, help="Fits a dipole to the map and removes it. Specify mask or type auto.",)
@click.option( "-remove_mono", default=None, type=click.STRING, help="Fits a monopole to the map and removes it.",)
@click.option( "-fontsize", default=None, type=click.INT, help="Fontsize",)
@click.option( "-figsize", default=None, help="Figure size",)
@click.option( "-darkmode", is_flag=True, help="Turn on darkmode",)
@click.option( "-sub", default=None, help="sub (ex. (2,1,2))",)
@click.option( "-hold", is_flag=True, help="Hold figure",)
@click.option( "-reuse_axes", is_flag=True, help="Reuse axes",)
@click.option( "-outdir", default=None, type=click.Path(exists=True), help="Output directory for plot",)
@click.option( "-outname", default=None, type=click.STRING, help="Output filename, overwrites autonaming",)
@click.option( "-show", is_flag=True, help="Displays the figure in addition to saving",)
@click.option( "-png", is_flag=True, help="Saves output as .png ().pdf by default)",)
@click.option( "-dpi", default=300, type=click.INT, help="DPI of output file. 300 by default.",)
# fmt: on
def gnom(
    input,
    **kwargs
):
    """
    Plots a fits file in gnomonic view.
    It is wrapper on the cosmoglobe "gnom" function.s
    """
    kwargs["cbar"] = not kwargs.pop("nocbar")
    kwargs["freq"] = None if kwargs["freq"] is None else kwargs["freq"] * u.GHz
    kwargs["fwhm"] = None if kwargs["fwhm"] is None else kwargs["fwhm"] * u.arcmin
    kwargs["figsize"] = (3,4) if kwargs["figsize"] is None else tuple(kwargs["figsize"])
    
    # Pop click-specific arguments
    show = kwargs.pop("show")
    dpi = kwargs.pop("dpi")
    png = kwargs.pop("png")
    outdir = kwargs.pop("outdir")
    outname = kwargs.pop("outname")

    img, params = cgnom(
        input,
        **kwargs
    )

    if show:
        plt.show()
    else:
        *path, fn = os.path.split(input)
        if outname:
            filetype = "png" if outname.endswith(".png") else "pdf"
        else:
            fn = os.path.splitext(fn)[0] + f'_gnomonic_{["I","Q","U"][kwargs["sig"]]}_{kwargs["lon"]}lon{kwargs["lat"]}lat_{kwargs["size"]}x{kwargs["size"]}deg_c-{params["cmap"]}'
            filetype = "png" if png else "pdf"

        if outname:
            fn = outname
        if outdir:
            path = [outdir]
        path = "." if path[0] == "" else path[0]
        fn = os.path.splitext(fn)[0] + f".{filetype}"
        print(f"[bold green]Outputting {fn}[/bold green] to {path}")
        plt.savefig(f"{path}/{fn}", bbox_inches="tight", pad_inches=0.02, format=filetype, dpi=dpi)

# fmt: off
@commands_plotting.command()
@click.argument( "input",)
@click.option( "-dataset", default=None, type=click.STRING, help="HDF dataset to be plotted",)
@click.option( "-sig", default=0, type=click.INT, help="Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)",)
@click.option( "-labels", default=None, help="List of label values for each input",)
@click.option( "-hideval", is_flag=True, help="Hide mean and standard deviation of samples",)
@click.option( "-burnin", default=0, type=click.INT, help="Burn in",)
@click.option( "-xlabel", default=None, type=click.STRING, help="set x axis label",)
@click.option( "-ylabel", default=None, type=click.STRING, help="set y axis label",)
@click.option( "-nbins", default=None, type=click.INT, help="Number of bins",)
@click.option( "-cmap", default="tab10", type=click.STRING, help="Color map (tab10 by default)",)
@click.option( "-figsize", default=None, help="figure size",)
@click.option( "-darkmode", is_flag=True, help="Turn on darkmode",)
@click.option( "-sub", default=None, help="sub (ex. (2,1,2))",)
@click.option( "-hold", is_flag=True, help="Hold figure",)
@click.option( "-reuse_axes", is_flag=True, help="Reuse axes",)
@click.option( "-outdir", default=None, type=click.Path(exists=True), help="Output directory for plot",)
@click.option( "-outname", default=None, type=click.STRING, help="Output filename, overwrites autonaming",)
@click.option( "-show", is_flag=True, help="Displays the figure in addition to saving",)
@click.option( "-png", is_flag=True, help="Saves output as .png ().pdf by default)",)
@click.option( "-dpi", default=300, type=click.INT, help="DPI of output file. 300 by default.",)
# fmt: on
def trace(
    input,
    **kwargs,
):
    """
    Creates a trace plot off data in a HDF file.
    This is a click wrapper the same function in cosmoglobe.
    """
    kwargs["cbar"] = not kwargs.pop("nocbar")
    kwargs["freq"] = None if kwargs["freq"] is None else kwargs["freq"] * u.GHz
    kwargs["fwhm"] = None if kwargs["fwhm"] is None else kwargs["fwhm"] * u.arcmin
    kwargs["figsize"] = (8,3) if kwargs["figsize"] is None else tuple(kwargs["figsize"])
    
    # Pop click-specific arguments
    show = kwargs.pop("show")
    dpi = kwargs.pop("dpi")
    png = kwargs.pop("png")
    outdir = kwargs.pop("outdir")
    outname = kwargs.pop("outname")

    kwargs["labels"] = kwargs["labels"].split(" ")
    kwargs["showval"] = not kwargs.pop("hideval")

    ctrace(
        input,
        **kwargs,
    )

    if show:
        plt.show()
    else:
        *path, fn = os.path.split(input)
        if outname:
            filetype = "png" if outname.endswith(".png") else "pdf"
        else:
            filetype = "png" if png else "pdf"

        if outname:
            fn = outname
        if outdir:
            path = [outdir]
        path = "." if path[0] == "" else path[0]
        fn = os.path.splitext(fn)[0] + f".{filetype}"
        print(f"[bold green]Outputting {fn}[/bold green] to {path}")
        plt.savefig(f"{path}/{fn}", bbox_inches="tight", pad_inches=0.02, format=filetype, dpi=dpi)
