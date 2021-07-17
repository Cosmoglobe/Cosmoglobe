from copy import Error
from multiprocessing import Value
from cosmoglobe.utils.utils import ModelError
from logging import error
from cosmoglobe.sky.model import Model

import warnings
import os
import healpy as hp
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from .plottools import *

# Fix for macos openMP duplicate bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@u.quantity_input(freq=u.Hz, fwhm=(u.arcmin, u.rad, u.deg))
def plot(
    input,
    sig=0,
    comp=None,
    freq=None,
    ticks=None,
    min=None,
    max=None,
    cbar=True,
    unit=None,
    fwhm=0.0 * u.arcmin,
    nside=None,
    mask=None,
    cmap=None,
    norm=None,
    remove_dip=False,
    remove_mono=False,
    right_label=None,
    left_label=None,
    width=None,
    xsize=1000,
    darkmode=False,
    interactive=False,
    rot=None,
    coord=None,
    nest=False,
    flip="astro",
    graticule=False,
    graticule_labels=False,
    return_only_data=False,
    projection_type="mollweide",
    cb_orientation="horizontal",
    xlabel=None,
    ylabel=None,
    longitude_grid_spacing=60,
    latitude_grid_spacing=30,
    override_plot_properties=None,
    xtick_label_color="black",
    ytick_label_color="black",
    graticule_color=None,
    fontsize=None,
    phi_convention="counterclockwise",
    custom_xtick_labels=None,
    custom_ytick_labels=None,
    **kwargs,
):
    """
    General plotting function for healpix maps.
    This function is a wrapper on healpys projview function with some added features.
    Features added in addition to existing projview features include:
        - direct plotting from model object, fits file or numpy array.
        - simple map operations such as smoothing, ud_grading, masking
        - removing dipole or monopoles
        - Predefined plotting parameters for specific components for nice formatting

    Parameters
    ----------
    input : ndarray, fits file path or cosmoglobe model object
        Map data input given as numpy array either 1d or index given by 'sig'.
        Also supports fits-file path string or cosmoglobe model.
        If cosmoglobe object is passed such as 'model', specify comp or freq.
    sig : str or int, optional
        Specify which signal to plot if ndim>1.
        default = None
    comp : string, optional
        Component label for automatic identification of plotting
        parameters based on information from autoparams.json 
        default = None
    freq : astropy GHz, optional
        frequency in GHz needed for scaling maps when using a model object input
        default = None
    min : float, optional
      The minimum range value. If specified, overwrites autodetector.
      default = None
    max : float, optional
      The maximum range value. If specified, overwrites autodetector.
      default = None
    ticks : list or str, optional
        Min and max value for data. If None, uses 97.5th percentile.
        default = None
    cbar : bool, optional
        Toggles the colorbar
        cbar = True
    fwhm : astropy arcmin/rad/deg, optional
        Optional map smoothing. FWHM of gaussian smoothing in arcmin.
        default = 0.0
    mask : str path or np.ndarray, optional
        Apply a mask file to data
        default = None
    remove_dip : bool, optional
        If mdmask is specified, fits and removes a dipole.
        default = True
    remove_mono : bool, optional
        If mdmask is specified, fits and removes a monopole.
        default = True
    norm : str, optional
        if norm=='linear':
            normal 
        if norm=='log':
            Normalizes data using a semi-logscale linear between -1 and 1.
            Autodetector uses this sometimes, you will be warned.
        default = None
    darkmode : bool, optional
        Plots all outlines in white for dark backgrounds, and adds 'dark' in
        filename.
        default = False
    cmap : str, optional
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default = None
    title : str, optional
        Sets the full figure title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    right_label : str, optional
        Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    left_label : str, optional
        Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default = None
    rot : scalar or sequence, optional
      Describe the rotation to apply.
      In the form (lon, lat, psi) (unit: degrees) : the point at
      longitude *lon* and latitude *lat* will be at the center. An additional rotation
      of angle *psi* around this direction is applied.
    coord : sequence of character, optional
      Either one of 'G', 'E' or 'C' to describe the coordinate
      system of the map, or a sequence of 2 of these to rotate
      the map from the first to the second coordinate system.
    nest : bool, optional
      If True, ordering scheme is NESTED. Default: False (RING)
    flip : {'astro', 'geo'}, optional
      Defines the convention of projection : 'astro' (default, east towards left, west towards right)
      or 'geo' (east towards roght, west towards left)
      It creates the `healpy_flip` attribute on the Axes to save the convention in the figure.
    graticule : bool
      add graticule
    graticule_labels : bool
      longitude and latitude labels
    projection_type :  {'aitoff', 'hammer', 'lambert', 'mollweide', 'cart', '3d', 'polar'}
      type of the plot
    cb_orientation : {'horizontal', 'vertical'}
      color bar orientation
    xlabel : str
      set x axis label
    ylabel : str
      set y axis label
    longitude_grid_spacing : float
      set x axis grid spacing
    latitude_grid_spacing : float
      set y axis grid spacing
    override_plot_properties : dict
      Override the following plot proporties: 'cbar_shrink', 'cbar_pad', 'cbar_label_pad', 'figure_width': width, 'figure_size_ratio': ratio.
    lcolor : str
      change the color of the longitude tick labels, some color maps make it hard to read black tick labels
    fontsize:  dict
        Override fontsize of labels: 'xlabel', 'ylabel', 'title', 'xtick_label', 'ytick_label', 'cbar_label', 'cbar_tick_label'.
    phi_convention : string
        convention on x-axis (phi), 'counterclockwise' (default), 'clockwise', 'symmetrical' (phi as it is truly given)
        if `flip` is 'geo', `phi_convention` should be set to 'clockwise'.
    custom_xtick_labels : list
        override x-axis tick labels
    custom_ytick_labels : list
        override y-axis tick labels
        kwargs : keywords
      any additional keyword is passed to pcolormesh
    """
    # Pick sizes from size dictionary for page width plots
    if width is None:
        override_plot_properties = None
    else:
        if isinstance(width, str):
            width = {"x": 2.75, "s": 3.5, "m": 4.7, "l": 7,}[width]
        ratio= 0.63 if cbar else 0.5
        xsize = int((1000 / 8.5) * width)
        override_plot_properties = {
            "figure_width": width,
            "figure_size_ratio": ratio,
        }

    set_style(darkmode)

    # Translate sig to correct format
    stokes = [
        "I",
        "Q",
        "U",
    ]
    if isinstance(sig, str):
        sig = stokes.index(sig)

    # Parsing component string
    comp_full = comp
    if comp is not None:
        comp, *specparam = comp.split()
    diffuse = True  # For future functionality

    # If map is string, interprate as file path
    if isinstance(input, str):
        # field = 0 if sig is None else sig
        m = hp.read_map(input, field=sig)
    elif isinstance(input, Model):
        """
        Get data from model object with frequency scaling        
        """
        if comp is None:
            if freq is not None:
                comp_full = "freqmap"
                m = input(freq, fwhm=fwhm,)
            else:
                raise ModelError(
                    f"Model object passed with comp and freq set to None"
                    f"comp: {comp} freq: {freq}"
                )
        else:
            if len(specparam) > 0 and isinstance(specparam[0], str):
                m = getattr(input, comp).spectral_parameters[specparam[0]]

                if len(m[sig]) == 1:
                    warnings.warn(
                        "Same value across the whole sky, mapping to array of length Npix"
                    )
                    m = np.full(hp.nside2npix(input.nside), m[sig])
            else:
                if freq is None:
                    freq_ref = getattr(input, comp).freq_ref
                    freq = freq_ref.value
                    m = getattr(input, comp).amp
                    if comp == "radio":
                        m = getattr(input, comp)(freq_ref, fwhm=fwhm)
                        diffuse = False
                    try:
                        freq = round(freq.squeeze()[sig], 5) * freq_ref.unit
                    except IndexError:
                        freq = round(freq, 5) * freq_ref.unit
                else:
                    m = getattr(input, comp)(freq, fwhm=fwhm)
    elif isinstance(input, np.ndarray):
        m = input
    else:
        raise TypeError(
            f"Type {type(input)} of input not supported"
            f"Supports numpy array, cosmoglobe model object or fits file string"
        )
    if not isinstance(input, Model) and freq is not None:
        freq = None
        warnings.warn("freq only usable with model object. Setting freq to None")
    # Make sure it is a 1d array
    if m.ndim > 1:
        m = m[sig]

    # Convert astropy map to numpy array
    if isinstance(m, u.Quantity):
        m = m.value

    # Mask map
    if mask is not None:
        if isinstance(mask, str):
            mask = hp.read_map(mask)

        m = hp.ma(m)
        m.mask = np.logical_not(mask)

    # ud_grade map
    if nside is not None and nside != hp.get_nside(m):
        m = hp.ud_grade(m, nside)
    else:
        nside = hp.get_nside(m)

    # Smooth map
    if fwhm > 0.0 and diffuse:
        m = hp.smoothing(m, fwhm.to(u.rad).value)

    # Remove mono/dipole
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    # Fetching autoset parameters
    params = autoparams(
        comp_full, sig, right_label, left_label, unit, ticks, min, max, norm, cmap, freq
    )

    # Ticks and ticklabels
    ticks = params["ticks"]
    if ticks == "auto" or params["norm"] == "hist":
        ticks = get_percentile(m, 97.5)
    elif None in ticks:
        pmin, pmax = get_percentile(m, 97.5)
        if ticks[0] is None:
            ticks[0] = pmin
        if ticks[-1] is None:
            ticks[-1] = pmax

    # Special case if dipole is detected in freqmap
    if comp_full == "freqmap":
        # if colorbar is super-saturated
        while len(m[abs(m) > ticks[-1]]) / hp.nside2npix(nside) > 0.7:
            ticks = [tick * 10 for tick in ticks]
            print("Colormap saturated. Expanding color-range.")

    # Create ticklabels from final ticks
    ticklabels = [fmt(i, 1) for i in ticks]

    # Semi-log normalization
    if params["norm"] == "log":
        m, ticks = apply_logscale(m, ticks, linthresh=1)

    # Colormap
    cmap = load_cmap(params["cmap"])

    # Plot using mollview if interactive mode
    if interactive:
        if params["norm"] == "log":
            params["norm"] = None
            # Ticklabels not available for this
            params["unit"] = r"$\log($" + params["unit"] + r"$)$"
        if right_label is None:
            ttl = params["right_title"]
        ret = hp.mollview(
            m,
            min=ticks[0],
            max=ticks[-1],
            cbar=cbar,
            cmap=cmap,
            unit=params["unit"],
            title=ttl,
            norm=params["norm"],
            # unedited params
            rot=rot,
            coord=coord,
            nest=nest,
            flip=flip,
            graticule=graticule,
            graticule_labels=graticule_labels,
            return_only_data=return_only_data,
            projection_type=projection_type,
            cb_orientation=cb_orientation,
            xlabel=xlabel,
            ylabel=ylabel,
            longitude_grid_spacing=longitude_grid_spacing,
            latitude_grid_spacing=latitude_grid_spacing,
            override_plot_properties=override_plot_properties,
            xtick_label_color=xtick_label_color,
            ytick_label_color=ytick_label_color,
            graticule_color=graticule_color,
            fontsize=fontsize,
            phi_convention=phi_convention,
            custom_xtick_labels=custom_xtick_labels,
            custom_ytick_labels=custom_ytick_labels,
        )
        return ret

    # Plot figure
    ret = hp.newvisufunc.projview(
        m,
        min=ticks[0],
        max=ticks[-1],
        cbar=False,
        cmap=cmap,
        xsize=xsize,
        override_plot_properties=override_plot_properties,
        # fontsize={
        #    'xlabel': 10,
        #    'ylabel': 10,
        #    'xtick_label': 10,
        #    'ytick_label': 10,
        #    'title': 10,
        #    'cbar_label': 10,
        #    'cbar_tick_label': 10,
        # },
        **kwargs,
    )

    # Remove color bar because of healpy bug
    plt.gca().collections[-1].colorbar.remove()
    # Add pretty color bar
    if cbar:
        apply_colorbar(
            plt.gcf(),
            plt.gca(),
            ret,
            ticks,
            ticklabels,
            params["unit"],
            linthresh=1,
            norm=params["norm"],
        )

    #### Right Title ####
    plt.text(
        4.5, 1.1, params["right_label"], ha="center", va="center",
    )
    #### Left Title (stokes parameter label by default) ####
    plt.text(
        -4.5, 1.1, params["left_label"], ha="center", va="center",
    )
    return ret


def apply_colorbar(fig, ax, image, ticks, ticklabels, unit, linthresh, norm=None):
    """
    This function applies a colorbar to the figure and formats the ticks.
    """
    from matplotlib.ticker import FuncFormatter

    cb = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        shrink=0.4,
        pad=0.04,
        ticks=ticks,
        format=FuncFormatter(fmt),
    )
    cb.ax.set_xticklabels(ticklabels,)
    cb.ax.xaxis.set_label_text(unit)
    if norm == "log":
        linticks = np.linspace(-1, 1, 3) * linthresh
        logmin = np.round(ticks[0])
        logmax = np.round(ticks[-1])

        logticks_min = -(10 ** np.arange(0, abs(logmin) + 1))
        logticks_max = 10 ** np.arange(0, logmax + 1)
        ticks_ = np.unique(np.concatenate((logticks_min, linticks, logticks_max)))
        # cb.set_ticks(np.concatenate((ticks,symlog(ticks_))), []) # Set major ticks

        logticks = symlog(ticks_, linthresh)
        logticks = [x for x in logticks if x not in ticks]
        cb.set_ticks(np.concatenate((ticks, logticks)))  # Set major ticks
        cb.ax.set_xticklabels(ticklabels + [""] * len(logticks))

        minorticks = np.linspace(-linthresh, linthresh, 5)
        minorticks2 = np.arange(2, 10) * linthresh

        for i in range(len(logticks_min)):
            minorticks = np.concatenate((-(10 ** i) * minorticks2, minorticks))
        for i in range(len(logticks_max)):
            minorticks = np.concatenate((minorticks, 10 ** i * minorticks2))

        minorticks = symlog(minorticks, linthresh)
        minorticks = minorticks[(minorticks >= ticks[0]) & (minorticks <= ticks[-1])]
        cb.ax.xaxis.set_ticks(minorticks, minor=True)

    cb.ax.tick_params(
        which="both", axis="x", direction="in",
    )
    cb.ax.xaxis.labelpad = 0
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    return cb
