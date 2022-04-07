from matplotlib import rcParams
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cycler import cycler

import healpy as hp
import matplotlib.pyplot as plt

from .plottools import *

from cosmoglobe.sky._units import Unit

# TODO:
# CO is currently hardcoded
# Make better custom band selection


def spec(
    model,
    pol=False,
    sky_fractions=(25, 85),
    xlim=(0.25, 4000),
    figsize=None,
    fraction=1,
    labelsize=11,
    comp_labelsize=9,
    band_labelsize=8,
    extend=True,
    darkmode=False,
    ame_polfrac=0.006,
    haslam=True,
    chipass=True,
    spass=True,
    cbass=True,
    quijote=False,
    wmap=True,
    planck=True,
    dirbe=True,
    litebird=False,
    custom_bands=None,
    include_co=True,
    add_error=True,
    fsky_pos=500,
    component_options=None,
    text_offset=2,
):
    """
    Generates RMS intensity plot for components in model.
        `model = cosmoglobe.model_from_chain(chain, nside=64, components=["synch", "ff", "ame", "dust", "radio"])`
        `spec(model)`

    Parameters
    ----------
    model : cosmoglobe Model object
        Object containing all sky components to be visualized
    pol : bool, optional
        Plot polarization instead of intensity
        default = None (temperature)
    sky_fractions : touple, optional
        Min and max sky fractions for thickness of lines
        default: (25,85)
    xlim : touple, optional
        The limits of the x axis
        default: (0.25, 4000)
    figsize : touple, optional
        size of figure. This is automatically set by extend.
        default : None
    fraction : float, optional
        Sets the figsize to be a fraction of a latex document
        default : None
    labelsize : float,
        Fontsize of ticklabels and axislabels
        default : 11
    comp_labelsize : float,
        Fontsize of component labels
        default : 9
    band_labelsize : float,
        Fontsize of frequency band labels
        default : 8
    extend: bool, optional
        Increases the xlimit and includes an additional subplot on top
        default: True
    darkmode : bool, optional
        Set True to turn all frame elements white for black background viewing.
        default: False
    ame_polfrac: float, optional
        Set the polarization fraction of AME with a scale factor.
        default: 0.006
    haslam : bool, optional
        Toggle this observation column
        default:  True
    chipass : bool, optional
        Toggle this observation column
        default:  True
    spass : bool, optional
        Toggle this observation column
        default:  True
    cbass : bool, optional
        Toggle this observation column
        default:  True
    quijote : bool, optional
        Toggle this observation column
        default:  False
    wmap : bool, optional
        Toggle this observation column
        default:  True
    planck : bool, optional
        Toggle this observation column
        default:  True
    dirbe : bool, optional
        Toggle this observation column
        default:  True
    litebird : bool, optional
        Toggle this observation column
        default:  False
    custom_bands : bool, optional
        TODO: NOT IMPLEMENTED, toggle specific bands
        default:  None
    include_co : bool,  optional
        Turn off CO lines
        default: True
    add_error : bool, optional
        Toggle added uncertainty to low S/N part of spectrum.
        default: True
    fsky_pos : float, optional
        Frequency position of f_sky labels
        default : 900
    component_options : dict, optional
        Allows for overriding dictionary that sets all the properties of components.
        Such as {"dust":{"position":30, "label": "Thing", "linestyle": ":"}}
        default : None
    text_offset : float, optional
        Text offset from lines
        default : 5
    """
    if fraction is not None:
        figsize = get_figure_width(fraction=fraction)
    set_style(darkmode, font="serif")

    # rcParams.update(figparams)
    black = "k"
    if darkmode:
        brokecol = "white"
        grey = "#C0C0C0"
    else:
        brokecol = black
        grey = "#A3A3A3"

    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
        grey,
    ]
    rcParams["axes.prop_cycle"] = cycler(color=colors)
    rcParams["ytick.right"] = True

    blue, red, green, purple, orange, teal, lightred, lightgreen, pink, yellow, grey = (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
    )

    # plt.rcParams.update(plt.rcParamsDefault)

    # Plot polarization or temperature
    sig = 1 if pol else 0

    # Set x and y limits if extended and pol
    if xlim == (0.25, 4000):
        if not extend:
            xlim = (9, 1500)
    xmin, xmax = xlim
    ymin, ymax = (0.05, 7e2) if not pol else (1.001e-3, 2e2)
    ymin2, ymax2 = (ymax + 100, 1e7)

    figsize = get_figure_width(fraction=fraction) if figsize is None else figsize

    if extend:
        ratio = 5  # Make subplot on top at 5:1 ratio
        fig, (ax2, ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(figsize[0], figsize[1]),
            gridspec_kw={"height_ratios": [1, ratio]},
        )
        aspect_ratio = figsize[0] / figsize[1] * 1.25  # Correct for ratio

        ax2.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax2.tick_params(labelbottom=False)
        ax2.xaxis.set_ticks_position("none")

        # ---- Adding broken axis lines ----
        d = 0.005  # how big to make the diagonal lines in axes coordinates

        kwargs = dict(transform=ax2.transAxes, color=brokecol, clip_on=False, linewidth=1)
        ax2.plot((-d, +d), (-d * ratio, +d * ratio), **kwargs)  # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d * ratio, +d * ratio), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    else:
        ymax2 = ymax
        ymin2 = ymax
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        aspect_ratio = figsize[0] / figsize[1]

    # Spectrum parameters
    N = 1000
    nu = np.logspace(np.log10(0.1), np.log10(5000), N)
    seds = seds_from_model(nu, model, pol=pol, sky_fractions=sky_fractions)

    # Get parameters for each component line
    foregrounds = get_foregrounds(pol, extend)
    if component_options is not None:
        # Loop over nested dictionary for optional arguments
        for comp_ in component_options:
            foregrounds[comp_].update(component_options[comp_])

    # Looping over foregrounds and calculating spectra
    i = 0
    for comp in foregrounds.keys():
        if comp.startswith("co") and include_co:  # get closest thing to ref freq
            foregrounds[comp]["params"][2], line_idx = find_nearest(nu, foregrounds[comp]["params"][2])
            foregrounds[comp]["spectrum"] = np.zeros((2, len(sky_fractions), N))
            foregrounds[comp]["spectrum"][sig][0][line_idx] = foregrounds[comp]["params"][0]
            foregrounds[comp]["spectrum"][sig][1][line_idx] = foregrounds[comp]["params"][1]

        if comp.startswith("bb"):
            a = 0.67 * 1e-1 if comp.endswith("2") else 0.67 * 1e-2
            sed = np.zeros((2, len(sky_fractions), N))
            cmb_blackbody = (np.ones(len(nu)) * Unit("uK_CMB")).to("uK_RJ", equivalencies=cmb_equivalencies(nu * u.GHz))
            sed[1] = a * cmb_blackbody
            foregrounds[comp]["spectrum"] = sed

        if comp in seds.keys():
            foregrounds[comp]["spectrum"] = seds[comp]
        else:
            continue

        if pol and comp == "ame":
            foregrounds[comp]["spectrum"][1] = ame_polfrac * foregrounds[comp]["spectrum"][0]

        if add_error and not comp.startswith("co") and not comp.startswith("bb") and not comp.startswith("cmb"):
            thresh = 0.1
            alpha = 0.5
            foregrounds[comp]["spectrum"][sig][0] = foregrounds[comp]["spectrum"][sig][0] * (1 - np.exp(-(abs(foregrounds[comp]["spectrum"][sig][0] / thresh) ** alpha)))
            foregrounds[comp]["spectrum"][sig][1] = foregrounds[comp]["spectrum"][sig][1] / (1 - np.exp(-(abs(foregrounds[comp]["spectrum"][sig][1] / thresh) ** alpha)))
            foregrounds[comp]["spectrum"][sig][0][np.isnan(foregrounds[comp]["spectrum"][sig][0])] = 0
            foregrounds[comp]["spectrum"][sig][1][np.isnan(foregrounds[comp]["spectrum"][sig][1])] = 0

        if foregrounds[comp]["sum"] and foregrounds[comp]["spectrum"] is not None:
            if i == 0:
                foregrounds["sumfg"]["spectrum"] = foregrounds[comp]["spectrum"].copy()
            else:
                foregrounds["sumfg"]["spectrum"] += foregrounds[comp]["spectrum"]
            i += 1

    # ---- Plotting foregrounds and labels ----
    j = 0
    for comp, params in foregrounds.items():  # Plot all fgs except sumf
        if params["spectrum"] is None and not comp.startswith("co"):
            continue
        if params["gradient"]:
            k = 1
            gradient_fill_between(
                ax,
                nu,
                params["spectrum"][sig][1] * 1e-2,
                params["spectrum"][sig][1],
                color=params["color"],
            )
        else:
            if comp == "sumfg":
                ax.loglog(
                    nu,
                    params["spectrum"][sig][1],
                    linestyle=params["linestyle"],
                    linewidth=1.5,
                    color=params["color"],
                )
                if extend:
                    ax2.loglog(
                        nu,
                        params["spectrum"][sig][1],
                        linestyle=params["linestyle"],
                        linewidth=1.5,
                        color=params["color"],
                    )
                k = 1
                try:
                    ax.loglog(
                        nu,
                        params["spectrum"][sig][0],
                        linestyle=params["linestyle"],
                        linewidth=1.5,
                        color=params["color"],
                    )
                    if extend:
                        ax2.loglog(
                            nu,
                            params["spectrum"][sig][0],
                            linestyle=params["linestyle"],
                            linewidth=1.5,
                            color=params["color"],
                        )
                    k = 1
                except:
                    pass
            elif comp.startswith("co"):
                if include_co:
                    ax.loglog(
                        [params["params"][2], params["params"][2]],
                        [
                            max(params["spectrum"][sig][0]),
                            max(params["spectrum"][sig][1]),
                        ],
                        linestyle=params["linestyle"],
                        linewidth=2,
                        color=params["color"],
                        zorder=1000,
                    )
                    k = 1
                else:
                    continue
            else:
                if comp == "cmb":
                    ax.loglog(
                        nu,
                        params["spectrum"][sig][0],
                        linestyle=params["linestyle"],
                        linewidth=2,
                        color=params["color"],
                    )
                    if extend:
                        ax2.loglog(
                            nu,
                            params["spectrum"][sig][0],
                            linestyle=params["linestyle"],
                            linewidth=4,
                            color=params["color"],
                        )
                    k = 0
                else:
                    ax.fill_between(
                        nu,
                        params["spectrum"][sig][1],
                        params["spectrum"][sig][0],
                        color=params["color"],
                        alpha=0.8,
                    )
                    if extend:
                        ax2.fill_between(
                            nu,
                            params["spectrum"][sig][1],
                            params["spectrum"][sig][0],
                            color=params["color"],
                            alpha=0.8,
                        )
                    k = 1

        if comp.startswith("co") and include_co:
            ax.text(
                foregrounds[comp]["params"][2],
                np.max(params["spectrum"][sig][k]) * 0.5,
                params["label"],
                color=params["color"],
                alpha=0.7,
                ha="right",
                va="center",
                rotation=90,
                fontsize=comp_labelsize,
                path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3))],
                zorder=1000,
            )
        else:
            # Label each component with rotation
            x0, y0, rotation = rotate_label(
                nu,
                params["spectrum"][sig][k],
                params["position"],
                (xmin, xmax),
                (ymin, ymax),
                aspect_ratio,
            )
            _text_offset = text_offset * 1.5 if comp == "sumfg" else text_offset
            ax.annotate(
                params["label"],
                xy=(x0, y0),
                xytext=(0, _text_offset),
                textcoords="offset pixels",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=comp_labelsize,
                color=params["color"],
                path_effects=[
                    path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3)),
                ],
                horizontalalignment="center",
            )

            if comp == "dust":
                # Label the sky-fraction around the dust line
                x0, y0, rotation = rotate_label(
                    nu,
                    params["spectrum"][sig][1],
                    fsky_pos,
                    (xmin, xmax),
                    (ymin, ymax),
                    aspect_ratio,
                )
                ax.annotate(
                    r"$f_{\mathrm{sky}}=" + "{:d}".format(int(sky_fractions[1])) + "\%$",
                    rotation=rotation,
                    rotation_mode="anchor",
                    xy=(x0, y0),
                    ha="center",
                    va="bottom",
                    fontsize=comp_labelsize,
                    color=grey,
                    xytext=(0, text_offset),
                    textcoords="offset pixels",
                    path_effects=[
                        path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3)),
                    ],
                )

                x0, y0, rotation = rotate_label(
                    nu,
                    params["spectrum"][sig][0],
                    fsky_pos,
                    (xmin, xmax),
                    (ymin, ymax),
                    aspect_ratio,
                )
                ax.annotate(
                    r"$f_{\mathrm{sky}}=" + "{:d}".format(int(sky_fractions[0])) + "\%$",
                    rotation=rotation,
                    rotation_mode="anchor",
                    xy=(x0, y0),
                    ha="center",
                    va="top",
                    fontsize=comp_labelsize,
                    color=grey,
                    xytext=(0, -text_offset),
                    textcoords="offset pixels",
                    path_effects=[
                        path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3)),
                    ],
                )

    # ---- Data band ranges ----
    if extend:
        yscaletext = 0.70
        yscaletextup = 1.2
    else:
        yscaletextup = 1.03
        yscaletext = 0.90

    if custom_bands is not None:
        """
        TODO: Make it easier to specify which bands to include
        """
        pass
    if isinstance(planck, bool):
        planck = [planck] * 9
    if isinstance(wmap, bool):
        wmap = [wmap] * 5

    # TODO add these as args?
    # fmt: off
    databands = {
        "Haslam": {
            "0.408\nHaslam": { "pol": False, "show": haslam, "position": [0.408, ymin * yscaletextup], "range": [0.406, 0.410], "color": purple, }
        },
        "S-PASS": {
            "2.303\nS-PASS": { "pol": True, "show": spass, "position": [2.35, ymax2 * yscaletext], "range": [2.1, 2.4], "color": green, }
        },
        "C-BASS": {
            "5.0\nC-BASS": { "pol": True, "show": cbass, "position": [5.0, ymax2 * yscaletext], "range": [4.0, 6.0], "color": blue, }
        },
        "CHI-PASS": {
            "1.394\nCHI-PASS": { "pol": False, "show": chipass, "position": [1.3945, ymin * yscaletextup], "range": [1.3945 - 0.064 / 2, 1.3945 + 0.064 / 2], "color": lightred, }
        },
        "QUIJOTE": {
            "11\nQUIJOTE": { "pol": True, "show": quijote, "position": [11, ymax2 * yscaletext], "range": [10.0, 12.0], "color": red,},
            "13": { "pol": True, "show": quijote, "position": [13, ymax2 * yscaletext], "range": [12.0, 14.0], "color": red,},
            "17": { "pol": True, "show": quijote, "position": [17, ymax2 * yscaletext], "range": [16.0, 18.0], "color": red,},
            "19": { "pol": True, "show": quijote, "position": [20, ymax2 * yscaletext], "range": [18.0, 21.0], "color": red,},
            ".\n31": { "pol": True, "show": quijote, "position": [31, ymax2 * yscaletext], "range": [26.0, 36.0], "color": red,},
            ".\n41": { "pol": True, "show": quijote, "position": [42, ymax2 * yscaletext], "range": [35.0, 47.0], "color": red,},
        },
        "Planck": {
            "30": { "pol": True, "show": planck[0], "position": [28, ymax2 * yscaletext], "range": [23.9, 34.5], "color": orange,},  # Planck 30
            "44": { "pol": True, "show": planck[1], "position": [44, ymax2 * yscaletext], "range": [39, 50], "color": orange,},  # Planck 44
            "70": { "pol": True, "show": planck[2], "position": [70, ymax2 * yscaletext], "range": [60, 78], "color": orange,},  # Planck 70
            "100": { "pol": True, "show": planck[3], "position": [100, ymax2 * yscaletext], "range": [82, 120], "color": orange,},  # Planck 100
            "143": { "pol": True, "show": planck[4], "position": [145, ymax2 * yscaletext], "range": [125, 170], "color": orange,},  # Planck 143
            "217": { "pol": True, "show": planck[5], "position": [217, ymax2 * yscaletext], "range": [180, 265], "color": orange,},  # Planck 217
            "353": { "pol": True, "show": planck[6], "position": [350, ymax2 * yscaletext], "range": [300, 430], "color": orange,},  # Planck 353
            "545": { "pol": False, "show": planck[7], "position": [540, ymax2 * yscaletext], "range": [450, 650], "color": orange,},  # Planck 545
            "857": { "pol": False, "show": planck[8], "position": [850, ymax2 * yscaletext], "range": [700, 1020], "color": orange,},
        },  # Planck 857
        "DIRBE": {
            "DIRBE\n1250": { "pol": False, "show": dirbe, "position": [1200, ymin * yscaletextup], "range": [1000, 1540], "color": red,},  # DIRBE 1250
            "2140": { "pol": False, "show": dirbe, "position": [1900, ymin * yscaletextup], "range": [1780, 2500], "color": red,},  # DIRBE 2140
            "3000": { "pol": False, "show": dirbe, "position": [3000, ymin * yscaletextup], "range": [2600, 3500], "color": red,},
        },  # DIRBE 3000
        "WMAP": {
            "K": { "pol": True, "show": wmap[0], "position": [23, ymin * yscaletextup], "range": [21, 25.5], "color": teal,},
            "Ka": { "pol": True, "show": wmap[1], "position": [33, ymin * yscaletextup], "range": [30, 37], "color": teal,},
            "Q": { "pol": True, "show": wmap[2], "position": [41.0, ymin * yscaletextup], "range": [38, 45], "color": teal,},
            "V": { "pol": True, "show": wmap[3], "position": [60.0, ymin * yscaletextup], "range": [54, 68], "color": teal,},
            "W": { "pol": True, "show": wmap[4], "position": [90.0, ymin * yscaletextup], "range": [84, 106], "color": teal,},
        },
        "LiteBIRD": {
            "40": { "pol": True, "show": litebird, "position": [40, ymax2 * yscaletext], "range": [34, 46], "color": red,},
            "50": { "pol": True, "show": litebird, "position": [50, ymax2 * yscaletext], "range": [43, 57], "color": red,},
            "60": { "pol": True, "show": litebird, "position": [60, ymax2 * yscaletext], "range": [53, 67], "color": red,},
            "68": { "pol": True, "show": litebird, "position": [68, ymax2 * yscaletext], "range": [60, 76], "color": red,},
            "78": { "pol": True, "show": litebird, "position": [78, ymax2 * yscaletext], "range": [69, 87], "color": red,},
            "89": { "pol": True, "show": litebird, "position": [89, ymax2 * yscaletext], "range": [79, 99], "color": red,},
            "100": { "pol": True, "show": litebird, "position": [100, ymax2 * yscaletext], "range": [89, 111], "color": red,},
            "119": { "pol": True, "show": litebird, "position": [119, ymax2 * yscaletext], "range": [101, 137], "color": red,},
            "140": { "pol": True, "show": litebird, "position": [140, ymax2 * yscaletext], "range": [119, 161], "color": red,},
            "166": { "pol": True, "show": litebird, "position": [166, ymax2 * yscaletext], "range": [141, 191], "color": red,},
            "195": { "pol": True, "show": litebird, "position": [195, ymax2 * yscaletext], "range": [165, 225], "color": red,},
            "235": { "pol": True, "show": litebird, "position": [235, ymax2 * yscaletext], "range": [200, 270], "color": red,},
            "280": { "pol": True, "show": litebird, "position": [280, ymax2 * yscaletext], "range": [238, 322], "color": red,},
            "337": { "pol": True, "show": litebird, "position": [337, ymax2 * yscaletext], "range": [287, 387], "color": red,},
            "402\nLiteBIRD": { "pol": True, "show": litebird, "position": [402, ymax2 * yscaletext], "range": [356, 458], "color": red,},
        },
    }
    # fmt: on
    
    # If band is not included, label another band:
    if planck[4]:
        oldkey = list(databands["Planck"].keys())[3]
    else:
        for i in [3, 2, 1, 5, 6, 7, 8]:
            if planck[i]:
                oldkey = list(databands["Planck"].keys())[i]
                break

    newkey = oldkey + "\nPlanck"
    databands["Planck"][newkey] = databands["Planck"].pop(oldkey)

    # Same for wmap
    if wmap[1]:
        oldkey = list(databands["WMAP"].keys())[1]
    else:
        for i in [1, 2, 0, 3, 4]:
            if planck[i]:
                oldkey = list(databands["Planck"].keys())[i]
                break

    newkey = "WMAP\n" + oldkey
    databands["WMAP"][newkey] = databands["WMAP"].pop(oldkey)

    # Set databands from dictonary
    for experiment, bands in databands.items():
        for label, band in bands.items():
            if band["show"]:
                # if label == "353" and not pol: continue # SPECIFIC FOR BP SETUP
                if pol and not band["pol"]:
                    continue  # Skip non-polarization bands
                if band["position"][0] >= xmax or band["position"][0] <= xmin:
                    continue  # Skip databands outside range
                va = "bottom" if experiment in ["WMAP", "CHI-PASS", "DIRBE", "Haslam"] else "top"  # VA for WMAP on bottom
                ha = (
                    "center"
                    if experiment
                    in [
                        "Planck",
                        "WMAP",
                        "DIRBE",
                    ]
                    else "center"
                )
                ax.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=-20, label=experiment)
                if extend:
                    ax2.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=-20, label=experiment)
                    if experiment in ["WMAP", "CHI-PASS", "DIRBE", "Haslam"]:
                        ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=band_labelsize, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3))])
                    else:
                        ax2.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=band_labelsize, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3))])
                else:
                    ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=band_labelsize, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(0.3, -0.3))])

    # ---- Axis stuff ----
    xticks = []
    xticks_ = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    for i, xtick in enumerate(xticks_):
        if xtick >= xmin and xtick <= xmax:
            xticks.append(xtick)

    yticks = []
    yticks_ = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for i, ytick in enumerate(yticks_):
        if ytick >= ymin and ytick <= ymax:
            yticks.append(ytick)

    ax.set(
        xscale="log",
        yscale="log",
        ylim=(ymin, ymax),
        xlim=(xmin, xmax),
        yticks=yticks,
        yticklabels=yticks,
        xticks=xticks,
        xticklabels=xticks,
    )
    formatter = ticker.FuncFormatter(fmt)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", which="both", labelsize=labelsize, direction="in")
    ax.tick_params(
        axis="y",
        labelrotation=90,
        labelsize=labelsize,
    )

    if extend:
        ax2.set(
            xscale="log",
            yscale="log",
            ylim=(ymin2, ymax2),
            xlim=(xmin, xmax),
            xticks=xticks,
            xticklabels=xticks,
        )
        ax2.yaxis.set_major_formatter(formatter)
        ax2.tick_params(axis="both", which="both", labelsize=labelsize, direction="in")
        ax2.tick_params(
            axis="y",
            labelrotation=90,
            labelsize=labelsize,
        )
        ax2.set_yticklabels(
            [fmt(x, 1) for x in ax2.get_yticks()],
            fontsize=labelsize,
            va="center",
        )

    # Why do i have to do this?
    ax.set_yticklabels(
        [fmt(x, 1) for x in ax.get_yticks()],
        fontsize=labelsize,
        va="center",
    )
    ax.set_xticklabels([fmt(x, 1) for x in ax.get_xticks()], fontsize=labelsize)

    # Axis labels
    sax = fig.add_subplot(111, frameon=False)
    sax.tick_params(
        labelcolor="none",
        top=False,
        bottom=False,
        left=True,
        right=False,
        width=0.0,
    )

    sax.set_ylabel(r"$\mathrm{RMS\ amplitude\ [}\mu\mathrm{K}_{\mathrm{RJ}}\mathrm{]}$", fontsize=labelsize, labelpad=0)
    sax.set_xlabel(r"$\mathrm{Frequency\ [GHz]}$", fontsize=labelsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.02)


def rotate_label(x, y, pos, xrange, yrange, aspect_ratio):
    """
    This function takes a line and calculates position and rotation
    for a label on top of it.
    Takes into account logspace etc.
    """
    x0, idx1 = find_nearest(x, pos)
    idx2 = idx1 + 2
    x1 = x[idx2]
    y0 = y[idx1]
    y1 = y[idx2]
    datascaling = np.log(xrange[0] / xrange[1]) / np.log(yrange[0] / yrange[1])
    rotator = datascaling / aspect_ratio
    alpha = np.arctan(np.log(y1 / y0) / np.log(x1 / x0) * rotator)
    rotation = np.rad2deg(alpha)  # *rotator
    return x0, y0, rotation


def get_foregrounds(pol, extend):
    """
    This function grabs a dictionary of the default component values
    used for the spectrum plotting style.
    Override this by passing the component_option dictionary to spec
    """
    if pol:
        p = 0.6 if extend else 15
        sd = 2 if extend else 70
        return {
            "synch": {
                "label": "Synchrotron",
                "params": [],
                "position": 20,
                "color": "C2",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "dust": {
                "label": "Thermal Dust",
                "params": [],
                "position": 250,
                "color": "C1",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "sumfg": {
                "label": "Sum fg.",
                "params": [],
                "position": 70,
                "color": "C10",
                "sum": False,
                "linestyle": "--",
                "gradient": False,
                "spectrum": None,
            },
            "bb-2": {
                "label": r"BB $r=10^{-2}$",
                "params": [
                    0.01,
                    "BB",
                ],
                "position": p,
                "color": "C10",
                "sum": False,
                "linestyle": "dotted",
                "gradient": True,
                "spectrum": None,
            },
            "bb-4": {
                "label": r"BB $r=10^{-4}$",
                "params": [
                    1e-4,
                    "BB",
                ],
                "position": p,
                "color": "C10",
                "sum": False,
                "linestyle": "dotted",
                "gradient": True,
                "spectrum": None,
            },
            "cmb": {
                "label": "CMB EE",
                "params": [1, "EE"],
                "position": p,
                "color": "C5",
                "sum": False,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "ame": {
                "label": "Spinning Dust",
                "params": [],
                "position": sd,
                "color": "C4",
                "sum": True,
                "linestyle": "solid",
                "gradient": True,
                "spectrum": None,
            },
        }
    else:
        p = 3 if extend else 65
        td = 10 if extend else 17
        return {
            "dust": {
                "label": "Thermal Dust",
                "params": [],
                "position": td,
                "color": "C1",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "ff": {
                "label": "Free-Free",
                "params": [],
                "position": 50,
                "color": "C0",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "ame": {
                "label": "Spinning Dust",
                "params": [],
                "position": p,
                "color": "C4",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "synch": {
                "label": "Synchrotron",
                "params": [],
                "position": 170,
                "color": "C2",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "co10": {
                "label": r"CO$_{1\rightarrow 0}$",
                "params": [0.5, 8, 115, 11.06],
                "position": p,
                "color": "C9",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "co21": {
                "label": r"CO$_{2\rightarrow 1}$",
                "params": [0.3, 5, 230.0, 14.01],
                "position": p,
                "color": "C9",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "co32": {
                "label": r"CO$_{3\rightarrow 2}$",
                "params": [0.3, 1, 345.0, 12.24],
                "position": p,
                "color": "C9",
                "sum": True,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
            "sumfg": {
                "label": "Sum fg.",
                "params": [],
                "position": 25,
                "color": "C10",
                "sum": False,
                "linestyle": "--",
                "gradient": False,
                "spectrum": None,
            },
            "cmb": {
                "label": "CMB",
                "params": [1.0, "TT"],
                "position": 70,
                "color": "C5",
                "sum": False,
                "linestyle": "solid",
                "gradient": False,
                "spectrum": None,
            },
        }
