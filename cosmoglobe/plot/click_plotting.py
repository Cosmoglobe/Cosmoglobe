import click
from rich import print
import astropy.units as u
import matplotlib.pyplot as plt
import os

import cosmoglobe.plot as splot
from cosmoglobe.sky import model_from_chain
from .plottools import *

@click.group()
def commands_plotting():
    pass

@commands_plotting.command()
@click.argument("input", nargs=-1)
@click.option("-sig", default=0, help="Signal to be plotted 0 by default (0, 1, 2 is interprated as IQU)",)
@click.option("-comp", default=None, type=click.STRING, help="Component name for autosetting plottingparameters",)
@click.option("-freq", default=None, type=click.FLOAT, help="Frequency in GHz needed for scaling maps when using a model object input",)
@click.option("-ticks", default=None, help="Min and max value for data. If None, uses 97.5th percentile.",)
@click.option("-min", default=None, type=click.FLOAT, help="Min value of colorbar, overrides autodetector.",)
@click.option("-max", default=None, type=click.FLOAT, help="Max value of colorbar, overrides autodetector.",)
@click.option("-nocbar", is_flag=True, help='Adds colorbar ("cb" in filename)',)
@click.option("-unit", default=None, type=click.STRING, help="Set unit (Under color bar), has LaTeX functionality. Ex. $\mu$",)
@click.option("-fwhm", default=0.0, type=click.FLOAT, help="FWHM of smoothing, in arcmin.",)
@click.option("-nside", default=None, type=click.INT, help="nside for optional ud_grade.",)
@click.option("-mask", default=None, type=click.STRING, help="Masks input with specified maskfile.",)
@click.option("-maskfill", default=None, type=click.STRING, help='Color to fill masked area. for example "gray". Transparent by default.',)
@click.option("-cmap", default=None, type=click.STRING, help="Chose colormap (ex. sunburst, planck, etc). Available are matplotlib and cmasher. Also qualitative plotly [ex. q-Plotly-4 (q for qualitative 4 for max color)]",)
@click.option("-norm", default=None, help='"Linear" is normal, "log" Normalizesusingsemi-logscale',)
@click.option("-remove_dip", is_flag=True, help="Fits a dipole to the map and removes it.",)
@click.option("-remove_mono", is_flag=True, help="Fits a monopole to the map and removes it.",)
@click.option("-title", default=None, type=click.STRING, help="Sets the full figure title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-right_label", default=None, type=click.STRING, help="Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-left_label", default=None, type=click.STRING, help="Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)",)
@click.option("-width", default=None, help="Size: in inches OR 1/3, 1/2 and full page width (8.8/12/18cm) [ex. x, s, m or l, or ex. slm for all], m by default",)
@click.option("-xsize", default=1000, type=click.INT, help="figuresize in px",)
@click.option("-darkmode", is_flag=True, help='Plots all outlines in white for dark bakgrounds ("dark" in filename)',)
@click.option("-interactive", is_flag=True, help='Uses healpys interactive mode, zooming is possible etc.',)
@click.option("-rot", default=None, help="Describe the rotation to apply. In the form (lon, lat, psi) (unit: degrees) : the point at longitude *lon* and latitude *lat* will be at the center. An additional rotation of angle *psi* around this direction is applied.",)
@click.option("-coord", default=None, help="Either one of 'G', 'E' or 'C' to describe the coordinate system of the map,",)
@click.option("-nest", is_flag=True, help="If True, ordering scheme is NESTED.",)
@click.option("-flip", default="astro", type=click.STRING, help="Defines the convention of projection : 'astro' or 'geo'",)
@click.option("-graticule", is_flag=True, help='Adds graticule',)
@click.option("-graticule_labels", is_flag=True, help="longitude and latitude labels",)
@click.option("-return_only_data", is_flag=True, help="Returns figure",)
@click.option("-projection_type", default="mollweide", type=click.STRING, help='{"aitoff", "hammer", "lambert", "mollweide", "cart", "3d", "polar"} type of the plot',)
@click.option("-cb_orientation", default="horizontal", type=click.STRING, help='{"horizontal", "vertical"} color bar orientation',)
@click.option("-xlabel", default=None, type=click.STRING, help='set x axis label',)
@click.option("-ylabel", default=None, type=click.STRING, help='set y axis label',)
@click.option("-longitude_grid_spacing", default=60, type=click.FLOAT, help="set x axis grid spacing",)
@click.option("-latitude_grid_spacing", default=30, type=click.FLOAT, help="set y axis grid spacing",)
@click.option("-override_plot_properties", default=None, help='Override the following plot proporties: "cbar_shrink", "cbar_pad", "cbar_label_pad", "figure_width": width, "figure_size_ratio": ratio.',)
@click.option("-xtick_label_color", default="black", type=click.STRING, help='color of longitude ticks',)
@click.option("-ytick_label_color", default="black", type=click.STRING, help='color of latitude ticks',)
@click.option("-graticule_color", default=None, type=click.STRING, help='color of latitude ticks',)
@click.option("-fontsize", default=None, help="Dictionary with fontsizes: 'xlabel', 'ylabel', 'title', 'xtick_label', 'ytick_label', 'cbar_label', 'cbar_tick_label'.",)
@click.option("-phi_convention", default="counterclockwise", type=click.STRING, help='convention on x-axis (phi), "counterclockwise" (default), "clockwise", "symmetrical" (phi as it is truly given) if "flip" is "geo", "phi_convention" should be set to "clockwise".',)
@click.option("-custom_xtick_labels", default=None, help="override x-axis tick labels",)
@click.option("-custom_ytick_labels", default=None, help="override y-axis tick labels",)
@click.option("-white_background", is_flag=True, help="Sets the background to be white. (Transparent by default [recommended])",)
@click.option("-png", is_flag=True, help="Saves output as .png ().pdf by default)",)
@click.option("-outdir", default=None, type=click.Path(exists=True), help="Output directory for plot",)
@click.option("-outname", default=None, type=click.STRING, help="Output filename, overwrites autonaming",)
@click.option("-show", is_flag=True, help='Displays the figure in addition to saving',)
@click.option("-gif", is_flag=True, help='Convert inputs to gif',)
@click.option("-fps", default=3, type=click.INT, help="Gif speed, 300 by default.",)
@click.option("-sample", default=-1, type=click.INT, help="Select sample when inputing .h5 file",)
def plot(input, sig, comp, freq, ticks, min, max, nocbar, unit, fwhm, nside, mask, maskfill, cmap, norm, remove_dip, remove_mono, title, right_label, left_label, width, xsize, darkmode, interactive, rot, coord, nest, flip, graticule, graticule_labels, return_only_data, projection_type, cb_orientation, xlabel, ylabel, longitude_grid_spacing, latitude_grid_spacing, override_plot_properties, xtick_label_color, ytick_label_color, graticule_color, fontsize, phi_convention, custom_xtick_labels, custom_ytick_labels, white_background, png, outdir, outname, show, gif, fps, sample):
    """
    This function invokes the plot function from the command line
    input: filename of fits or h5 file, optional plotting paramameters
    """

    imgs = [] # Container for images used with gif
    for i, filename in enumerate(input):
        cbar = False if nocbar else True
        freq_ = None if freq is None else freq*u.GHz
        fwhm_ = None if fwhm is None else fwhm*u.arcmin
        comps=comp 
        if ".h5" in filename:
            if sample is None:
                raise ValueError(
                    "HDF file passed, please specify sample"
                )
            if comp is None:
                if freq is None:             
                    print("[bold orange]Warning! Neither frequency nor component selected. Plotting sky at 70GHz[/bold orange]")
                    freq=70*u.GHz
                comp = "freqmap"
            filename = model_from_chain(filename, comps=comps, nside=nside, samples=sample)

        # Super hacky gif fix because "hold" is not implemented in newvisufunc
        if gif and i>0: return_only_data=True

        # Actually plot the stuff
        img, params = splot(filename, sig=sig, comp=comps, freq=freq_, ticks=ticks, min=min, max=max, cbar=cbar, unit=unit, fwhm=fwhm_, nside=nside, mask=mask, maskfill=maskfill, cmap=cmap, norm=norm, remove_dip=remove_dip, remove_mono=remove_mono, title=title, right_label=right_label, left_label=left_label, width=width, xsize=xsize, darkmode=darkmode, interactive=interactive, rot=rot, coord=coord, nest=nest, flip=flip, graticule=graticule, graticule_labels=graticule_labels, return_only_data=return_only_data, projection_type=projection_type, cb_orientation=cb_orientation, xlabel=xlabel, ylabel=ylabel, longitude_grid_spacing=longitude_grid_spacing, latitude_grid_spacing=latitude_grid_spacing, override_plot_properties=override_plot_properties, xtick_label_color=xtick_label_color, ytick_label_color=ytick_label_color, graticule_color=graticule_color, fontsize=fontsize, phi_convention=phi_convention, custom_xtick_labels=custom_xtick_labels, custom_ytick_labels=custom_ytick_labels,)

        # Super hacky gif fix because "hold" is not implemented in newvisufunc
        if gif and i>0: 
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
        if show:
            plt.show()

        """
        Filename formatting and  outputting
        """
        if ".h5" in filename:
            filename = filename.replace(os.path.split(filename)[-1],os.path.split(filename)[-1]+"_"+comp)
            filename = filename.replace(".h5","")
        else:
            filename = filename.replace(".fits", "")


        *path, outfile = os.path.split(filename)
        #if path==['']: path="."
        outfile = outfile.replace("_IQU_", "_").replace("_I_", "_")
        if freq is not None: outfile = outfile.replace(comp, f"{comp}-{int(freq.value)}GHz")
        if outdir: path = outdir

        filename = []
        filename.append(f"{str(int(fwhm_.value))}arcmin") if float(fwhm_.value) > 0 else None
        filename.append("cb") if cbar else None
        filename.append("masked") if mask else None
        filename.append("nodip") if remove_dip else None
        filename.append("dark") if darkmode else None
        if params["cmap"] is None: params["cmap"]="planck"
        filename.append(f'c-{params["cmap"]}')

        nside_tag = "_n" + str(int(params["nside"]))
        if nside_tag in outfile:
            outfile = outfile.replace(nside_tag, "")

        if params["width"] is None: width = 8.5

        if isinstance(sig, int): sig = ["I","Q", "U"][sig]
        fn = outfile + f'_{sig}_w{str(int(width))}' + nside_tag

        for i in filename:
            fn += f"_{i}"

        filetype = "png" if png else "pdf"
        if outname:
            filetype = "png" if outname.endswith(".png") else "pdf"

        fn += f".{filetype}"
        if outname: fn = outname

        if outdir:
            fn = outdir + "/" + os.path.split(fn)[-1]


        if gif:
            # Append images for animation 
            imgs.append([img])
        else:
            tp = False if white_background else True  
            filetype = "png" if png else "pdf"
            print(f"[bold green]Outputting {fn}[/bold green]")
            path = "." if path[0]=="" else path[0]
            plt.savefig(f"{path}/{fn}", bbox_inches="tight", pad_inches=0.02, transparent=tp, format=filetype, dpi=300)
    
    if gif:
        import matplotlib.animation as animation
        ani = animation.ArtistAnimation(plt.gcf(), imgs, blit=True)
        fn = fn.replace(filetype,"gif")
        print(f"[bold green]Outputting {fn}[/bold green]")
        ani.save(fn,dpi=300,writer=animation.PillowWriter(fps=fps))

if __name__ == '__main__':
    plot()