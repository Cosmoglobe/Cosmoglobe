import healpy as hp
import matplotlib.pyplot as plt

from functools import partial
from .plottools import *


@u.quantity_input(freq=u.Hz, fwhm=(u.arcmin, u.rad, u.deg))
def gnom(
    input,
    lon=0,
    lat=0,
    comp=None,
    sig=0,
    freq=None,
    nside=None,
    size=20,
    sample=-1,
    vmin=None,
    vmax=None,
    ticks=None,
    rng=None,
    left_label=None,
    right_label=None,
    unit=None,
    cmap=None,
    graticule=False,
    norm=None,
    cbar=True,
    cbar_pad=0.04,
    cbar_shrink=0.7,
    fwhm=0.0 * u.arcmin,
    remove_dip=False,
    remove_mono=False,
    fontsize=None,
    figsize=(3, 4),
    darkmode=False,
    sub=None,
    hold=False,
    reuse_axes=False,
):
    """
    Gnomonic view plotting

    Parameters
    ----------
    input : ndarray, fits file path or cosmoglobe model object
        Map data input given as numpy array either 1d or index given by 'sig'.
        Also supports fits-file path string or cosmoglobe model.
        If cosmoglobe object is passed such as 'model', specify comp or freq.
    lon : float
        Longitude coordinate of the center of the figure in galactic coordinates
        default: 0
    lat : float
        Latitude coordinate of the center of the figure in galactic coordinates
        default: 0
    sig : str or int, optional
        Specify which signal to plot if ndim>1.
        default: None
    comp : string, optional
        Component label for automatic identification of plotting
        parameters based on information from autoparams.json
        default: None
    freq : astropy GHz, optional
        frequency in GHz needed for scaling maps when using a model object input
        default: None
    nside : int, optional
        optional ud_grading of the input data
        default: None
    size : float, optional
        Size of square in degrees
        default: 20
    sample : float, optional
        Specify sample if passing chain
        default: -1
    vmin : float, optional
        Min value
        default: None
    vmax : float, optional
        Max value
        default: None
    ticks : list or str, optional
        Ticks for colorbar
        default: None
    rng : float, optional
      Sets this value as min and max value. If specified, overwrites autodetector.
      default: None
    left_label : str, optional
        Sets the upper left title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default: None
    right_label : str, optional
        Sets the upper right title. Has LaTeX functionaliity (ex. $A_{s}$.)
        default: None
    unit : str, optional
        Unit label for colorbar
        default: None
    cmap : str, optional
        Colormap (ex. sunburst, planck, jet). Both matplotliib and cmasher
        available as of now. Also supports qualitative plotly map, [ex.
        q-Plotly-4 (q for qualitative 4 for max color)] Sets planck as default.
        default: None
    graticule : bool
        add graticule
        default: False
    norm : str, matplotlib norm object, optional
        if norm=='linear':
            normal
        if norm=='log':
            Uses log10 scale
            Normalizes data using a semi-logscale linear between -1 and 1.
            Autodetector uses this sometimes, you will be warned.
        SPECIFY linthresh for linear threshold of symlog!
        default: None
    cbar : bool, optional
        Toggles the colorbar
        cbar : True
    cbar_pad : float, optioinal
        default: 0.04    
    cbar_shrink : float, optional
        default: 0.7    
    fwhm : astropy arcmin/rad/deg, optional
        Optional map smoothing. FWHM of gaussian smoothing in arcmin.
        default: 0.0
    remove_dip : bool, optional
        If mdmask is specified, fits and removes a dipole.
        default: True
    remove_mono : bool, optional
        If mdmask is specified, fits and removes a monopole.
        default: True
    fontsize : int, optional
        Size of labels
        default: None
    figsize : touple, optional 
        size of figure
        default: None
    darkmode : bool, optional 
        turn all axis elements white for optimal dark visualization
        default: False
    sub : int, scalar or sequence, optional
        Use only a zone of the current figure (same syntax as subplot).
        Default: None
    hold : bool, optional
        If True, replace the current Axes by a MollweideAxes.
        use this if you want to have multiple maps on the same
        figure. 
        Default: False
    reuse_axes : bool, optional
        If True, reuse the current Axes (should be a MollweideAxes). This is
        useful if you want to overplot with a partially transparent colormap,
        such as for plotting a line integral convolution. 
        Default: False

    mask : str path or np.ndarray, optional
        Apply a mask file to data
        default = None
    """
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
    # Set plotting rcParams
    set_style(darkmode)
    xsize = 1000
    reso = size * 60 / xsize

    # Get data
    m, comp, freq, nside  = get_data(input, sig, comp, freq, fwhm, nside=nside, sample=sample)
    if remove_dip:
        m = hp.remove_dipole(m, gal_cut=30, copy=True, verbose=True)
    if remove_mono:
        m = hp.remove_monopole(m, gal_cut=30, copy=True, verbose=True)

    proj = hp.projector.GnomonicProj(
        rot=[lon, lat, 0.0], coord="G", xsize=xsize, ysize=xsize, reso=reso
    )
    reproj_im = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))

    # Pass all your arguments in, return parsed plotting parameters
    params = get_params(
        data=reproj_im, 
        comp=comp,
        sig=sig,
        right_label=right_label,
        left_label=left_label,
        unit=unit,
        ticks=ticks,
        min=vmin,
        max=vmax,
        rng=rng,
        norm=norm,
        cmap=cmap,
        freq_ref=freq,
        width=figsize[0],
        nside=nside
    )

    # Semi-log normalization
    if params["norm"] == "log":
        params["data"], params["ticks"] = apply_logscale(params["data"], params["ticks"], linthresh=1)

    cmap = load_cmap(params["cmap"])

    fig, ax = make_fig(
        figsize,
        None,
        hold,
        sub,
        reuse_axes
    )
    image = plt.imshow(
        params["data"],
        origin="lower",
        interpolation="nearest",
        vmin=params["ticks"][0],
        vmax=params["ticks"][-1],
        cmap=cmap
    )

    plt.xticks([])
    plt.yticks([])
    plt.text(
        0.95,
        0.95,
        params["right_label"],
        color="black",
        va="top",
        ha="right",
        transform=ax.transAxes,
        fontsize=fontsize["title"]
    )
    plt.text(
        0.05,
        0.95,
        params["left_label"],
        color="black",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize["title"]
    )

    if cbar:
        apply_colorbar(
            plt.gcf(),
            plt.gca(),
            image,
            params["ticks"],
            params["ticklabels"],
            params["unit"],
            fontsize=fontsize,
            linthresh=1,
            norm=params["norm"],
            cbar_pad=cbar_pad,
            cbar_shrink=cbar_shrink
        )

    if graticule:
        hp.graticule()

    return image, params
