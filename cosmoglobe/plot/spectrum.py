from matplotlib import rcParams
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cycler import cycler

import healpy as hp
import matplotlib.pyplot as plt

from .plottools import *

pol=False
long=True
if pol:
    # 15, 120, 40, (0,4, 12), (1.2,50)
    p = 0.6 if long else 15
    sd = 2 if long else 70
    foregrounds = {
        "synch" : {         "label"   : "Synchrotron",
                            "params"  : [],
                            "position": 20,
                            "color"   : "C2",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        },
        "dust": {           "label" : "Thermal Dust",
                            "params": [],
                            "position": 250,
                            "color":    "C1",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        }, 
        "sumfg"      : {    "label"   : "Sum fg.",
                            "params"  : [],
                            "position": 70,
                            "color"   : "grey",
                            "sum"     : False,
                            "linestyle": "--",
                            "gradient": False,
                            "spectrum": None,
                        },
        r"bb-2"   :  {"label"   : "BB $r=10^{-2}$", 
                            "params"  : [0.01, "BB",],
                            "position": p,
                            "color"   : "grey",
                            "sum"     : False,
                            "linestyle": "dotted",
                            "gradient": True,
                            "spectrum": None,
                        },
        r"bb-4$"   :  {"label"   : "BB $r=10^{-4}$", 
                            "params"  : [1e-4, "BB",],
                            "position": p,
                            "color"   : "grey",
                            "sum"     : False,
                            "linestyle": "dotted",
                            "gradient": True,
                            "spectrum": None,
                        },
        "cmb":       {"label"     : "CMB EE", 
                            "params"  : [1, "EE"],
                            "position": p,
                            "color"   : "C5",
                            "sum"     : False,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        },
        "ame" : {"label"    : "Spinning Dust", 
                            "params"  : [],
                            "position": sd,
                            "color"   : "C4",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": True,
                            "spectrum": None,
                        },

        }
else:
    #120, 12, 40, (2,57), 20, 70
    p = 3 if long else 65
    td = 10 if long else 17
    foregrounds = {
        "synch" : {"label"      : "Synchrotron", 
                            "params"  : [],
                            "position": 170,
                            "color"   : "C2",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        },
        "dust": {"label"      : "Thermal Dust", 
                            "params"  : [],
                            "position": td,
                            "color"   :  "C1",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        }, 
        "ff"  : {"label"       : "Free-Free", 
                            "params"  : [],
                            "position": 50,
                            "color"   : "C0",
                            "sum"     : True,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        },
        "ame" : {"label"     : "Spinning Dust", 
                            "params"   : [],
                            "position" : p,
                            "color"    : "C4",
                            "sum"      : True,
                            "linestyle": "solid",
                            "gradient" : False,
                            "spectrum": None,
                        },
        r"co10": {"label"    : "CO$_{1\rightarrow 0}$", 
                                    "params"  : [50, 115, 11.06],
                                    "position": p,
                                    "color"   : "C9",
                                    "sum"     : True,
                                    "linestyle": "solid",
                                    "gradient": False,
                                    "spectrum": None,
                        },
        r"co21": {"label"    : "CO$_{2\rightarrow 1}$", 
                                    "params"  : [25, 230., 14.01],
                                    "position": p,
                                    "color"   : "C9",
                                    "sum"     : True,
                                    "linestyle": "solid",
                                    "gradient": False,
                                    "spectrum": None,
                        },
        r"co32":      {"label"     : "CO$_{3\rightarrow 2}$", 
                                        "params"  : [10, 345., 12.24],
                                        "position": p,
                                        "color"   : "C9",
                                        "sum"     : True,
                                        "linestyle": "solid",
                                        "gradient": False,
                                        "spectrum": None,
                        },
        "sumfg"      : {"label"     : "Sum fg.", 
                            "params"  : [],
                            "position": 25,
                            "color"   : "grey",
                            "sum"     : False,
                            "linestyle": "--",
                            "gradient": False,
                            "spectrum": None,
                        },
        "cmb":          {"label"     : "CMB", 
                            "params"  : [1., "TT"],
                            "position": 70,
                            "color"   : "C5",
                            "sum"     : False,
                            "linestyle": "solid",
                            "gradient": False,
                            "spectrum": None,
                        },

        }

"""
    Spectrum(pol, long, darkmode, png, foregrounds, [mask1,mask2], nside)

        # ---- Figure parameters ----
    if not small:
        xmin, xmax = (0.25, 4000)
    if pol:
        ymin, ymax = (1.001e-3, 2e2)
        if not small:
            #xmin, xmax = (1, 3000)
            ymin2, ymax2 = (ymax+100, 1e7)
        else:
            xmin, xmax = (9, 1500)
    else:
        ymin, ymax = (0.05, 7e2)
        if not small:
            #xmin, xmax = (0.3, 4000)
            ymin2, ymax2 = (ymax+500, 1e7)
        else:
            xmin, xmax = (9, 1500)
"""

def spec(model, pol=False, nside=64, sky_fractions=(25,85), darkmode=False):
    # TODO, they need to be smoothed to common res!

    rcParams['mathtext.fontset'] = 'dejavusans'
    colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    rcParams['axes.prop_cycle'] = cycler(color=colors)
    blue, red, green, purple, orange, teal, lightred, lightgreen, pink, yellow = ("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9",)
    black = 'k'
    #plt.rcParams.update(plt.rcParamsDefault)
    #set_style(darkmode)

    pol = False
    small = False
    pol = 1 if pol else 0
    xmin, xmax = (0.25, 4000) if not small else (9, 1500)
    ymin, ymax = (0.05, 7e2) if not pol else (1.001e-3, 2e2)
    ymin2, ymax2 = (ymax+100, 1e7)

    if not small:    
        # Figure
        ratio = 5
        w, h = (16,8)
        fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(w,h),gridspec_kw = {'height_ratios':[1, ratio]})
        aspect_ratio = w/h*1.25 # Correct for ratio

        ax2.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.tick_params(labelbottom=False)
        ax2.xaxis.set_ticks_position('none')

        # ---- Adding broken axis lines ----
        d = .005  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax2.transAxes, color=black, clip_on=False)
        ax2.plot((-d, +d), (-d*ratio, + d*ratio), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d*ratio, +d*ratio), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
        # textsize
        freqtext = 16
        fgtext = 18

    else:
        ymax2=ymax
        ymin2=ymax
        w, h = (12,8)
        fig, ax = plt.subplots(1,1,figsize=(w,h))
        aspect_ratio = w/h
        
        freqtext = 20
        fgtext = 20

    
    # Spectrum parameters
    nu  = np.logspace(np.log10(0.1),np.log10(5000),1000)
    #nside=model.nside
    npix = hp.nside2npix(nside)
    sky_fractions=(25,85,100)
    seds = seds_from_model(nu, model, pol=True, sky_fractions=sky_fractions)


    # Looping over foregrounds and calculating spectra
    i = 0
    add_error = True
    for comp in foregrounds.keys():
        if comp in seds.keys():
            foregrounds[comp]["spectrum"] = seds[comp]

        # TODO Fix errorextension
        """
        if add_error and not comp.startswith("CO"):
            thresh=0.1                    
            alpha=0.5
            foregrounds[comp]["spectrum"][0] = foregrounds[comp]["spectrum"][0]*(1-np.exp(-(abs(foregrounds[comp]["spectrum"][0]/thresh)**alpha)))
            foregrounds[comp]["spectrum"][1] = foregrounds[comp]["spectrum"][1]/(1-np.exp(-(abs(foregrounds[comp]["spectrum"][1]/thresh)**alpha)))
        """

        if comp.startswith("co"): # get closest thing to ref freq
            foregrounds[comp]["params"][-2], _ = find_nearest(nu, foregrounds[comp]["params"][-2])


        if foregrounds[comp]["sum"] and foregrounds[comp]["spectrum"] is not None:
            if i==0:
                foregrounds["sumfg"]["spectrum"] = foregrounds[comp]["spectrum"]
            else:
                foregrounds["sumfg"]["spectrum"] += foregrounds[comp]["spectrum"]
            i+=1

    # ---- Plotting foregrounds and labels ----
    j=0
    for comp, params in foregrounds.items(): # Plot all fgs except sumf
        if params["spectrum"] is None: continue
        if params["gradient"]:
            k = 1
            gradient_fill_between(ax, nu, params["spectrum"][pol][0], params["spectrum"][pol][1], color=params["color"])
        else:
            if comp == "sumfg":
                ax.loglog(nu,params["spectrum"][pol][0], linestyle=params["linestyle"], linewidth=2, color=params["color"])
                if not small:
                    ax2.loglog(nu,params["spectrum"][pol][0], linestyle=params["linestyle"], linewidth=2, color=params["color"])
                k = 0
                """
                try:
                    ax.loglog(nu,params["spectrum"][pol][1], linestyle=params["linestyle"], linewidth=2, color=params["color"])
                    if not small:
                        ax2.loglog(nu,params["spectrum"][pol][1], linestyle=params["linestyle"], linewidth=2, color=params["color"])
                    k=1
                except:
                    pass
                """
            elif comp.startswith("co"):
                lfreq = nu[np.argmax(params["spectrum"][pol][0])]
                if params["spectrum"][pol].shape[0] > 1:
                    ax.loglog([lfreq,lfreq],[max(params["spectrum"][pol][0]), max(params["spectrum"][pol][1])], linestyle=params["linestyle"], linewidth=4, color=params["color"],zorder=1000)
                    k=1
                else:
                    k=0
                    ax.bar(lfreq, params["spectrum"][pol][0], color=black,)
            else:
                if comp == "cmb":
                    ax.loglog(nu,params["spectrum"][pol][0], linestyle=params["linestyle"], linewidth=4, color=params["color"])
                    if not small:
                        ax2.loglog(nu,params["spectrum"][pol][0], linestyle=params["linestyle"], linewidth=4, color=params["color"])
                    k = 0
                else:
                    ax.fill_between(nu,params["spectrum"][pol][0],params["spectrum"][pol][1], color=params["color"],alpha=0.5)
                    if not small:
                        ax2.fill_between(nu,params["spectrum"][pol][0],params["spectrum"][pol][1], color=params["color"], alpha=0.5)
                    k = 1

        if comp == "dust":
            _, fsky_idx = find_nearest(nu, 900)
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(sky_fractions[1])), xy=(nu[fsky_idx], params["spectrum"][pol][1][fsky_idx]), ha="center", va="bottom", fontsize=fgtext, color="grey", xytext=(0,5), textcoords="offset pixels",)
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(sky_fractions[0])), xy=(nu[fsky_idx], params["spectrum"][pol][0][fsky_idx]), ha="center", va="top", fontsize=fgtext, color="grey", xytext=(0,-15), textcoords="offset pixels",)
       
        if comp.startswith("co"):
            ax.text(lfreq, np.max(params["spectrum"][pol][k])*0.5, params["label"], color=params["color"], alpha=0.7, ha='right',va='center',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))], zorder=1000)
        else:
            x0, idx1 = find_nearest(nu, params["position"])
            idx2 = idx1+2
            x1 = nu[idx2] 
            # idx2 = find_nearest(nu, params["position"]**1.05)
            y0 = params["spectrum"][pol][k][idx1]
            y1 = params["spectrum"][pol][k][idx2]
            datascaling  = np.log(xmin/xmax)/np.log(ymin/ymax)
            rotator = (datascaling/aspect_ratio)
            alpha = np.arctan(np.log(y1/y0)/np.log(x1/x0)*rotator)
            rotation =  np.rad2deg(alpha)#*rotator
            ax.annotate(params["label"], xy=(x0,y0), xytext=(0,7), textcoords="offset pixels",  rotation=rotation, rotation_mode='anchor', fontsize=fgtext, color=params["color"], path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(1, -1)),], horizontalalignment="center")
    
        
    
    # ---- Data band ranges ----
    if not small:
        yscaletext = 0.70
        yscaletextup = 1.2
    else:
        yscaletextup = 1.03
        yscaletext = 0.90

    # TODO add these as args?
    haslam = True
    chipass = True
    spass = True
    cbass = True
    quijote = True
    wmap = True
    planck = True
    dirbe = True
    databands = {"Haslam":  {"0.408\nHaslam": {"pol": False, "show": haslam, "position": [.408, ymin*yscaletextup],  "range": [.406,.410], "color": purple,}},
                 "S-PASS":  {"2.303\nS-PASS":  {"pol": True, "show": spass,  "position": [2.35, ymax2*yscaletext],  "range": [2.1,2.4], "color": green,}},
                 "C-BASS":  {"5.0\nC-BASS":   {"pol": True, "show": cbass,  "position": [5., ymax2*yscaletext],    "range": [4.,6.], "color": blue,}},
                 "CHI-PASS":{"1.394\nCHI-PASS":{"pol": False, "show": chipass,"position": [1.3945, ymin*yscaletextup],"range": [1.3945-0.064/2, 1.3945+0.064/2], "color": lightred,}},
                 "QUIJOTE": {"11\nQUIJOTE":    {"pol": True, "show": quijote,"position": [11, ymax2*yscaletext],    "range":  [10.,12.], "color": red,},
                             "13":             {"pol": True, "show": quijote, "position": [13, ymax2*yscaletext], "range":  [12.,14.], "color": red,},
                             "17":             {"pol": True, "show": quijote, "position": [17, ymax2*yscaletext], "range":  [16.,18.], "color": red,},
                             "19":             {"pol": True, "show": quijote, "position": [20, ymax2*yscaletext], "range":  [18.,21.], "color": red,},
                             ".\n31":             {"pol": True, "show": quijote, "position": [31, ymax2*yscaletext], "range":  [26.,36.], "color": red,},
                             ".\n41":             {"pol": True, "show": quijote, "position": [42, ymax2*yscaletext], "range":  [35.,47.], "color": red,}},
                 "Planck":  {"30":          {"pol": True, "show": planck, "position": [27,  ymax2*yscaletext], "range": [23.9,34.5],"color": orange,},      # Planck 30
                             "44":          {"pol": True, "show": planck, "position": [40,  ymax2*yscaletext], "range": [39,50]    ,"color": orange,},      # Planck 44
                             "70":          {"pol": True, "show": planck, "position": [60,  ymax2*yscaletext], "range": [60,78]    ,"color": orange,},      # Planck 70
                             "100\nPlanck": {"pol": True, "show": planck, "position": [90,  ymax2*yscaletext], "range": [82,120]   ,"color": orange,},      # Planck 100
                             "143":         {"pol": True, "show": planck, "position": [130, ymax2*yscaletext], "range": [125,170]  ,"color": orange,},      # Planck 143
                             "217":         {"pol": True, "show": planck, "position": [195, ymax2*yscaletext], "range": [180,265]  ,"color": orange,},      # Planck 217
                             "353":         {"pol": True, "show": planck, "position": [320, ymax2*yscaletext], "range": [300,430]  ,"color": orange,},      # Planck 353
                             "545":         {"pol": False, "show": planck, "position": [490, ymax2*yscaletext], "range": [450,650]  ,"color": orange,},      # Planck 545
                             "857":         {"pol": False, "show": planck, "position": [730, ymax2*yscaletext], "range": [700,1020] ,"color": orange,}},      # Planck 857
                 "DIRBE":   {"DIRBE\n1250":  {"pol": False, "show": dirbe, "position": [1000, ymin*yscaletextup], "range": [1000,1540] , "color": red,},     # DIRBE 1250
                             "2140":         {"pol": False, "show": dirbe, "position": [1750, ymin*yscaletextup], "range": [1780,2500] , "color": red,},     # DIRBE 2140
                             "3000":         {"pol": False, "show": dirbe, "position": [2500, ymin*yscaletextup], "range": [2600,3500] , "color": red,}},     # DIRBE 3000
                 "WMAP":    {"K": {"pol": True, "show": wmap, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": teal,}, 
                             "WMAP\nKa":      {"pol": True, "show": wmap, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": teal,},
                             "Q":       {"pol": True, "show": wmap, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": teal,}, 
                             "V":       {"pol": True, "show": wmap, "position": [58.,  ymin*yscaletextup], "range": [54,68], "color": teal,}, 
                             "W":       {"pol": True, "show": wmap, "position": [90.,  ymin*yscaletextup], "range": [84,106], "color": teal,}}, 
    }
    # Set databands from dictonary
    for experiment, bands in databands.items():
        for label, band in bands.items():
            if band["show"]:
                #if label == "353" and not pol: continue # SPECIFIC FOR BP SETUP
                if pol and not band["pol"]:
                    continue # Skip non-polarization bands
                if band["position"][0]>=xmax or band["position"][0]<=xmin:
                    continue # Skip databands outside range
                va = "bottom" if experiment in ["WMAP", "CHI-PASS", "DIRBE", "Haslam"] else "top" # VA for WMAP on bottom
                ha = "left" if experiment in ["Planck", "WMAP", "DIRBE",] else "center"
                ax.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=0, label=experiment)
                if not small:
                    ax2.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=0, label=experiment)
                    if experiment in  ["WMAP", "CHI-PASS", "DIRBE", "Haslam"]:
                        ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                    else:
                        ax2.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                else:
                    ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])

    # ---- Axis stuff ----
    lsize=20

    ticks = []
    ticks_ = [0.3,1,3,10,30,100,300,1000,3000]

    for i, tick in enumerate(ticks_):
        if tick>=xmin and tick<=xmax:
            ticks.append(tick)
    ax.set(xscale='log', yscale='log', ylim=(ymin, ymax), xlim=(xmin,xmax),xticks=ticks, xticklabels=ticks)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=lsize, direction='in')
    ax.tick_params(which="both",direction="in")
    ax.tick_params(axis='y', labelrotation=90) 
    ax.set_yticklabels([fmt(x,1) for x in ax.get_yticks()], va="center")
    if not small:
        ax2.set(xscale='log', yscale='log', ylim=(ymin2, ymax2), xlim=(xmin,xmax), yticks=[1e4,1e6,], xticks=ticks, xticklabels=ticks)
        ax2.tick_params(axis='both', which='major', labelsize=lsize, direction='in')
        ax2.tick_params(which="both",direction="in")
        ax2.tick_params(axis='y', labelrotation=90,) 
        ax2.set_yticklabels([fmt(x,1) for x in ax2.get_yticks()], va="center")

    # Axis labels
    if pol:
        plt.ylabel(r"RMS polarization amplitude [$\mu\mathrm{K}_{\mathrm{RJ}}$]",fontsize=lsize)
    else:
        plt.ylabel(r"RMS brightness temperature [$\mu\mathrm{K}_{\mathrm{RJ}}$]",fontsize=lsize)
    plt.xlabel(r"Frequency [GHz]",fontsize=lsize)

