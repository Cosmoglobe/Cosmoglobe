from copy import Error
from multiprocessing import Value
from logging import error

import warnings
import os
from rich import print
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
    sample=-1,
    mask=None,
    maskfill=None,
    cmap=None,
    norm=None,
    remove_dip=False,
    remove_mono=False,
    title=None,
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
    ticks : list or str, optional
        Min and max value for data. If None, uses 97.5th percentile.
        default = None
    min : float, optional
      The minimum range value. If specified, overwrites autodetector.
      default = None
    max : float, optional
      The maximum range value. If specified, overwrites autodetector.
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
    cmap : str, optional
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default = None
    norm : str, optional
        if norm=='linear':
            normal
        if norm=='log':
            Normalizes data using a semi-logscale linear between -1 and 1.
            Autodetector uses this sometimes, you will be warned.
        default = None
    remove_dip : bool, optional
        If mdmask is specified, fits and removes a dipole.
        default = True
    remove_mono : bool, optional
        If mdmask is specified, fits and removes a monopole.
        default = True
    unit : str, optional
        Unit label for colorbar
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
    darkmode : bool, optional
        Plots all outlines in white for dark backgrounds, and adds 'dark' in
        filename.
        default = False
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
    return_only_data : bool
      Return figure
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
        default = None
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
        try:
            width = int(width)
        except:
            if isinstance(width, str):
                width = {
                    "x": 2.75,
                    "s": 3.5,
                    "m": 4.7,
                    "l": 7,
                }[width]
        ratio = 0.63 if cbar else 0.5
        xsize = int((1000 / 8.5) * width)
        override_plot_properties = {
            "figure_width": width,
            "figure_size_ratio": ratio,
        }

    if not fontsize:
        fontsize = {
            "xlabel": 11,
            "ylabel": 11,
            "xtick_label": 8,
            "ytick_label": 8,
            "title": 12,
            "cbar_label": 11,
            "cbar_tick_label": 9,
            "left_label": 11,
            "right_label": 11,
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

    # Get data
    m = get_data(input, sig, comp, freq, fwhm, nside=nside, sample=sample)

    # ud_grade map
    if nside is not None and nside != hp.get_nside(m):
        m = hp.ud_grade(m, nside)
    else:
        nside = hp.get_nside(m)

    # Mask map
    if mask is not None:
        if isinstance(mask, str):
            mask = hp.read_map(mask)

        m = hp.ma(m)
        if hp.get_nside(mask) != nside:
            print("[orange]Input mask nside is different, ud_grading to output nside.[/orange]")
            mask = hp.ud_grade(mask, nside)
        m.mask = np.logical_not(mask)

    # Smooth map
    if fwhm > 0.0:
        m = hp.smoothing(m, fwhm.to(u.rad).value)

    # Remove mono/dipole
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    # Fetching autoset parameters
    params = autoparams(
        comp, sig, right_label, left_label, unit, ticks, min, max, norm, cmap, freq
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

    # Update parameter dictionary
    params["nside"] = nside
    params["width"] = width
    params["ticks"] = ticks

    # Special case if dipole is detected in freqmap
    if comp == "freqmap":
        # if colorbar is super-saturated
        while len(m[abs(m) > ticks[-1]]) / hp.nside2npix(nside) > 0.7:
            ticks = [tick * 10 for tick in ticks]
            print("[orange]Colormap saturated. Expanding color-range.[/orange]")

    # Create ticklabels from final ticks
    ticklabels = [fmt(i, 1) for i in ticks]

    # Semi-log normalization
    if params["norm"] == "log":
        m, ticks = apply_logscale(m, ticks, linthresh=1)

    # Colormap
    cmap = load_cmap(params["cmap"])
    if maskfill:
        cmap.set_bad(maskfill)

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
        return ret, params

    warnings.filterwarnings("ignore") # Healpy complains too much
    # Plot figure
    ret = hp.newvisufunc.projview(
        m,
        min=ticks[0],
        max=ticks[-1],
        cbar=False,
        cmap=cmap,
        xsize=xsize,
        # unedited params
        title=title,
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
        **kwargs,
    )
    if not return_only_data:
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
                fontsize=fontsize,
                linthresh=1,
                norm=params["norm"],
            )

    #### Right Title ####
    plt.text(
        0.925,
        0.925,
        params["right_label"],
        ha="center",
        va="center",
        fontsize=fontsize["right_label"],
        transform=plt.gca().transAxes,
    )
    #### Left Title (stokes parameter label by default) ####
    plt.text(
        0.075,
        0.925,
        params["left_label"],
        ha="center",
        va="center",
        fontsize=fontsize["left_label"],
        transform=plt.gca().transAxes,
    )
    return ret, params

