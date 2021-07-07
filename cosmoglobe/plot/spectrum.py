from matplotlib import rcParams, rc
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager
from cycler import cycler
from tqdm import trange, tqdm
import numpy as np
import healpy as hp
import sys
import math

import src.tools as tls


    from src.spectrum import Spectrum
    if pol:
        # 15, 120, 40, (0,4, 12), (1.2,50)
        p = 0.6 if long else 15
        sd = 2 if long else 70
        foregrounds = {
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s,],
                             "position": 20,
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
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "--",
                             "gradient": False,
                          },
            r"BB $r=10^{-2}$"   :  {"function": "rspectrum", 
                             "params"  : [0.01, "BB",],
                             "position": p,
                             "color"   : "grey",
                             "sum"     : False,
                             "linestyle": "dotted",
                             "gradient": True,
                         },
            r"BB $r=10^{-4}$"   :  {"function": "rspectrum", 
                             "params"  : [1e-4, "BB",],
                             "position": p,
                             "color"   : "grey",
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
        foregrounds = {
            "Synchrotron" : {"function": "lf", 
                             "params"  : [a_s, b_s,],
                             "position": 170,
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
                             "position": 50,
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
                             "position": 25,
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


def Spectrum(pol, small=False, darkmode, png, foregrounds, masks, nside, cmap=None):
    rcParams['mathtext.fontset'] = 'dejavusans'
    rcParams['axes.prop_cycle'] = cycler(color=getattr(pcol.qualitative, cmap))
    blue, red, green, purple, orange, teal, lightred, lightgreen, pink, yellow = ("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9",)
    black = 'k'
    

    # ---- Figure parameters ----
    if not small:
        xmin, xmax = (0.25, 4000)
    if pol:
        ymin, ymax = (1.001e-3, 2e2)
        if not small:
            #xmin, xmax = (1, 3000)
            ymax15, ymax2 = (ymax+100, 1e7)
        else:
            xmin, xmax = (9, 1500)
    else:
        ymin, ymax = (0.05, 7e2)
        if not small:
            #xmin, xmax = (0.3, 4000)
            ymax15, ymax2 = (ymax+500, 1e7)
        else:
            xmin, xmax = (9, 1500)

    if not small:    

        # Figure
        ratio = 5
        w, h = (16,8)
        fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(w,h),gridspec_kw = {'height_ratios':[1, ratio]})
        aspect_ratio = w/h*1.25 # Correct for ratio
        rotdir = -1

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
        ymax15=ymax
        w, h = (12,8)
        fig, ax = plt.subplots(1,1,figsize=(w,h))
        aspect_ratio = w/h
        rotdir = 1
        #ax.set_aspect('equal', adjustable='box')
        
        freqtext = 20
        fgtext = 20

    # Spectrum parameters
    field = 1 if pol else 0
    nu  = np.logspace(np.log10(0.1),np.log10(5000),1000)
    npix = hp.nside2npix(nside)
    # Read masks
    m = np.ones((len(masks), npix))
    for i, mask in enumerate(masks):
        # Read and ud_grade mask
        if mask:
            m_temp = hp.read_map(mask, field=0, dtype=None, verbose=False)
            if hp.npix2nside(len(m_temp)) != nside:
                m[i] = hp.ud_grade(m_temp, nside)
                m[i,m[i,:]>0.5] = 1 # Set all mask values to integer    
                m[i,m[i,:]<0.5] = 0 # Set all mask values to integer   
            else:
                m[i] = m_temp


    # Get indices of smallest mask
    idx = m[np.argmax(np.sum(m, axis=1)), :] > 0.5
    skyfracs = np.sum(m,axis=1)/npix*100
    print(f"Using sky fractions {skyfracs}%")
    # Looping over foregrounds and calculating spectra
    i = 0
    add_error = True
    for fg in foregrounds.keys():
        if not fg == "Sum fg.":
            if fg.startswith("CO"): # get closest thing to ref freq
                foregrounds[fg]["params"][-2], _ = find_nearest(nu, foregrounds[fg]["params"][-2])

            foregrounds[fg]["spectrum"] = getspec(nu*1e9, fg, foregrounds[fg]["params"], foregrounds[fg]["function"], field, nside, npix, idx, m,)
            foregrounds[fg]["spectrum_mean"]= np.mean(foregrounds[fg]["spectrum"],axis=0)
            if add_error and foregrounds[fg]["spectrum"].shape[0]>1 and not fg.startswith("CO"):
                thresh=0.1                    
                alpha=0.5
                foregrounds[fg]["spectrum"][0] = foregrounds[fg]["spectrum"][0]*(1-np.exp(-(abs(foregrounds[fg]["spectrum"][0]/thresh)**alpha)))
                foregrounds[fg]["spectrum"][1] = foregrounds[fg]["spectrum"][1]/(1-np.exp(-(abs(foregrounds[fg]["spectrum"][1]/thresh)**alpha)))

        if foregrounds[fg]["sum"]:
            if i==0:
                if foregrounds[fg]["spectrum"].shape[0] == 1:
                    # special case where first summed is 1d
                    foregrounds["Sum fg."]["spectrum"] = np.concatenate((foregrounds[fg]["spectrum"],foregrounds[fg]["spectrum"])).copy()
                else:
                    foregrounds["Sum fg."]["spectrum"] = foregrounds[fg]["spectrum"].copy()
            else:
                foregrounds["Sum fg."]["spectrum"] += foregrounds[fg]["spectrum"]
            i+=1

    # ---- Plotting foregrounds and labels ----
    j=0
    for label, fg in foregrounds.items(): # Plot all fgs except sumf
        if fg["gradient"]:
            if label == "Spinning Dust":
                k = 1
                gradient_fill_between(ax, nu, fg["spectrum"][0], fg["spectrum"][1], color=fg["color"])
            else:
                k = 0
                gradient_fill(nu, fg["spectrum"][k], fill_color=fg["color"], ax=ax, alpha=0.5, linewidth=0.0,)

        else:
            if label == "Sum fg.":
                ax.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                if not small:
                    ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                k = 0
                try:
                    ax.loglog(nu,fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    if not small:
                        ax2.loglog(nu,fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    k=1
                except:
                    pass
            elif label.startswith("CO"):
                lfreq = nu[np.argmax(fg["spectrum"][0])]
                if fg["spectrum"].shape[0] > 1:
                    ax.loglog([lfreq,lfreq],[max(fg["spectrum"][0]), max(fg["spectrum"][1])], linestyle=fg["linestyle"], linewidth=4, color=fg["color"],zorder=1000)
                    k=1
                else:
                    k=0
                    ax.bar(lfreq, fg["spectrum"][0], color=black,)
            else:
                if fg["spectrum"].shape[0] == 1:
                    ax.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                    if not small:
                        ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                    k = 0
                else:
                    #gradient_fill(nu, fg["spectrum"][0], fill_color=fg["color"], ax=ax, alpha=0.5, linewidth=0.0,)
                    
                    ax.loglog(nu,fg["spectrum_mean"], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                    ax.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], color=fg["color"],alpha=0.5)

                    if not small:
                        ax2.loglog(nu,fg["spectrum_mean"], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                        ax2.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], color=fg["color"], alpha=0.5)
                    k = 1

        if label == "Thermal Dust" and fg["spectrum"].shape[0]>1:
            _, fsky_idx = find_nearest(nu, 900)
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(skyfracs[1])), xy=(nu[fsky_idx], fg["spectrum"][1][fsky_idx]), ha="center", va="bottom", fontsize=fgtext, color="grey", xytext=(0,5), textcoords="offset pixels",)
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(skyfracs[0])), xy=(nu[fsky_idx], fg["spectrum"][0][fsky_idx]), ha="center", va="top", fontsize=fgtext, color="grey", xytext=(0,-15), textcoords="offset pixels",)
       
        if label.startswith("CO"):
            ax.text(lfreq, np.max(fg["spectrum"][k])*0.5, label, color=fg["color"], alpha=0.7, ha='right',va='center',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))], zorder=1000)
        else:
            x0, idx1 = find_nearest(nu, fg["position"])
            idx2 = idx1+2
            x1 = nu[idx2] 
            # idx2 = find_nearest(nu, fg["position"]**1.05)
            y0 = fg["spectrum"][k][idx1]
            y1 = fg["spectrum"][k][idx2]
            datascaling  = np.log(xmin/xmax)/np.log(ymin/ymax)
            rotator = (datascaling/aspect_ratio)
            alpha = np.arctan(np.log(y1/y0)/np.log(x1/x0)*rotator)
            rotation =  np.rad2deg(alpha)#*rotator
            ax.annotate(label, xy=(x0,y0), xytext=(0,7), textcoords="offset pixels",  rotation=rotation, rotation_mode='anchor', fontsize=fgtext, color=fg["color"], path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(1, -1)),], horizontalalignment="center")
    
        
    
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
        ax2.set(xscale='log', yscale='log', ylim=(ymax15, ymax2), xlim=(xmin,xmax), yticks=[1e4,1e6,], xticks=ticks, xticklabels=ticks)
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



def fmt(x, pos):
    """
    Format color bar labels
    """
    a, b = f"{x:.2e}".split("e")
    b = int(b)
    if float(a) == 1.00:
        return r"$10^{"+str(b)+"}$"
    elif float(a) == -1.00:
        return r"$-10^{"+str(b)+"}$"
    else:
        return fr"${a} \cdot 10^{b}$"

	
# This function calculates the intensity spectra
# Alternative 1 uses 2 masks to calculate spatial variations
# Alternative 2 uses only scalar values
def getspec(nu, fg, params, function, field, nside, npix, idx, m):
    val = []
    #print(fg)
    # Alternative 1
    if any([str(x).endswith(".fits") for x in params]) or any([isinstance(x,np.ndarray) for x in params]):
        if fg == "Spinning Dust":
            from pathlib import Path
            ame_template = Path(__file__).parent / "spdust2_cnm.dat"
            fnu, f_ = np.loadtxt(ame_template, unpack=True)
            fnu *= 1e9
            field = 0

        temp = []
        nsides = []
        # Read all maps and record nsides
        
        for i, p in enumerate(params):
            if str(p).endswith(".fits"):
                if field==1 and i==0: # If polarization amplitude map
                    s1 = hp.read_map(p, field=1, dtype=None, verbose=False)
                    s2 = hp.read_map(p, field=2, dtype=None, verbose=False)
                    p = np.sqrt(s1**2+s2**2)
                else:
                    p = hp.read_map(p, field=field, dtype=None, verbose=False)
                nsides.append(hp.npix2nside(len(p)))
            elif isinstance(p, np.ndarray):
                if not fg == "Spinning Dust":
                    if field==1 and i==0:
                        p = np.sqrt(p[1]**2+p[2]**2)
                    elif p.ndim > 1 and p.shape[0]>1:
                        p = p[field]
                nsides.append(hp.npix2nside(len(p)))
            else:
                nsides.append(0)
            temp.append(p)


        # Create dataset and convert to same resolution
        params = np.zeros(( len(params), npix ))
        for i, t in enumerate(temp):
            if nsides[i] == 0:
                params[i,:] = t
            elif nsides[i] != nside:
                params[i,:] = hp.ud_grade(t, nside)
            else:
                params[i,:] = t
        # Only calculate outside masked region    
        N = 1000
        map_ = np.zeros((N, npix))

        for i, nu_ in enumerate(tqdm(nu, desc = fg, ncols=80)):
            if fg == "Spinning Dust":
                map_[i, idx] = getattr(tls, function)(nu_, *params[:,idx], fnu, f_) #fgs.fg(nu, *params[pix])
            else:
                map_[i, idx] = getattr(tls, function)(nu_, *params[:,idx]) #fgs.fg(nu, *params[pix])

        # Apply mask to all frequency points
        # calculate mean 
        rmss = []
        for i in range(2):
            n = np.sum(m[i])            
            masked = hp.ma(map_)
            masked.mask = np.logical_not(m[i])
            mono = masked.mean(axis=1)
            masked -= mono.reshape(-1,1)
            rms = np.sqrt( ( (masked**2).sum(axis=1) ) /n)
            val.append(rms)

        vals = np.sort(np.array(val), axis=0) 
    else:
        # Alternative 2
        val = getattr(tls, function)(nu, *params) #fgs.fg(nu, *params))
        #vals = np.stack((val, val),)
        vals = val.reshape(1,-1)
    return vals
