from matplotlib import rcParams
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cycler import cycler
from tqdm import tqdm
import numpy as np
import healpy as hp

import cosmoglobe.release.tools as tls
def Spectrum(pol, long, darkmode, png, foregrounds, masks, nside, cmap="Plotly"):
    params = {'savefig.dpi'        : 300, # save figures to 300 dpi
              'xtick.top'          : False,
              'ytick.right'        : True, #Set to false
              'axes.spines.top'    : True, #Set to false
              'axes.spines.bottom' : True,
              'axes.spines.left'   : True,
              'axes.spines.right'  : True, #Set to false@
              'axes.grid.axis'     : 'y',
              'axes.grid'          : False,
              'ytick.major.size'   : 10,
              'ytick.minor.size'   : 5,
              'xtick.major.size'   : 10,
              'xtick.minor.size'   : 5,
              'xtick.major.pad'    : 8, 
              'ytick.major.width'   : 1.5,
              'ytick.minor.width'   : 1.5,
              'xtick.major.width'   : 1.5,
              'xtick.minor.width'   : 1.5,
              'axes.linewidth'      : 1.5,
              'mathtext.fontset': 'dejavusans',
              #'ytick.major.size'   : 6,
              #'ytick.minor.size'   : 3,
              #'xtick.major.size'   : 6,
              #'xtick.minor.size'   : 3,
              
    }
    black = 'k'
    if darkmode:
        brokecol="white"
        grey="#C0C0C0"
    else:
        brokecol=black
        grey="grey"
    colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', grey,]
    rcParams['axes.prop_cycle'] = cycler(color=colors)
    blue, red, green, purple, orange, teal, lightred, lightgreen, pink, yellow, grey = ("C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10")

    if darkmode:
        rcParams['text.color']   = 'white'   # axes background color
        rcParams['axes.facecolor']   = 'white'   # axes background color
        rcParams['axes.edgecolor' ]  = 'white'   # axes edge color
        rcParams['axes.labelcolor']   = 'white'
        rcParams['xtick.color']    = 'white'      # color of the tick labels
        rcParams['ytick.color']    = 'white'      # color of the tick labels
        rcParams['grid.color'] =   'white'   # grid color
        rcParams['legend.facecolor']    = 'inherit'   # legend background color (when 'inherit' uses axes.facecolor)
        rcParams['legend.edgecolor']= 'white' # legend edge color (when 'inherit' uses axes.edgecolor)
        black = 'white'
    rcParams.update(params)
    
    # ---- Figure parameters ----
    if long:
        xmin, xmax = (0.25, 4000)
    if pol:
        ymin, ymax = (1.001e-3, 2e2)
        if long:
            #xmin, xmax = (1, 3000)
            ymax15, ymax2 = (ymax+100, 1e7)
        else:
            xmin, xmax = (9, 1500)
    else:
        ymin, ymax = (0.05, 7e2)
        if long:
            #xmin, xmax = (0.3, 4000)
            ymax15, ymax2 = (ymax+500, 1e7)
        else:
            xmin, xmax = (9, 1500)

    if long:    

        # Figure
        ratio = 5
        w, h = (16,8)
        fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(w,h),gridspec_kw = {'height_ratios':[1, ratio]})

        ax2.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.tick_params(labelbottom=False)
        ax2.xaxis.set_ticks_position('none')

        # ---- Adding broken axis lines ----
        d = .005  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax2.transAxes, color=brokecol, clip_on=False)
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
    print("Foreground items:", foregrounds.items())
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
        print(f"plotting {label}")
        if fg["gradient"]:
            k = fg["spectrum"].shape[0] - 1
            gradient_fill_between(ax, nu, fg["spectrum"][k]*1e-2, fg["spectrum"][k], color=fg["color"])
        else:
            if label == "Sum fg.":
                ax.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                if long:
                    ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                k = 0
                try:
                    ax.loglog(nu,fg["spectrum"][1], linestyle=fg["linestyle"], linewidth=2, color=fg["color"])
                    if long:
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
                    if long:
                        ax2.loglog(nu,fg["spectrum"][0], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                    k = 0
                else:
                    #gradient_fill(nu, fg["spectrum"][0], fill_color=fg["color"], ax=ax, alpha=0.5, linewidth=0.0,)
                    
                    #ax.loglog(nu,fg["spectrum_mean"], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                    ax.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], color=fg["color"],alpha=0.8)

                    if long:
                        #ax2.loglog(nu,fg["spectrum_mean"], linestyle=fg["linestyle"], linewidth=4, color=fg["color"])
                        ax2.fill_between(nu,fg["spectrum"][0],fg["spectrum"][1], color=fg["color"], alpha=0.8)
                    k = 1

        if label == "Thermal Dust" and fg["spectrum"].shape[0]>1:
            _, fsky_idx = find_nearest(nu, 900)
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(skyfracs[1])), xy=(nu[fsky_idx], fg["spectrum"][1][fsky_idx]), ha="center", va="bottom", fontsize=fgtext, color="grey", xytext=(0,5), textcoords="offset pixels",path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(0.5, -0.5)),])
            ax.annotate(r"$f_{sky}=$"+"{:d}%".format(int(skyfracs[0])), xy=(nu[fsky_idx], fg["spectrum"][0][fsky_idx]), ha="center", va="top", fontsize=fgtext, color="grey", xytext=(0,-15), textcoords="offset pixels",path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(0.5, -0.5)),])
       
        if label.startswith("CO"):
            ax.text(lfreq, np.max(fg["spectrum"][k])*0.5, label, color=fg["color"], alpha=0.7, ha='right',va='center',rotation=90,fontsize=fgtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))], zorder=1000)
        else:
            print(label, fg["position"])
            x0, idx1 = find_nearest(nu, fg["position"])
            idx2 = idx1+2
            x1 = nu[idx2] 
            # idx2 = find_nearest(nu, fg["position"]**1.05)
            y0 = fg["spectrum"][k][idx1]
            y1 = fg["spectrum"][k][idx2]
            if y0>ymax:
                aspect_ratio = w/(h/6) if long else w/h
                datascaling  = np.log(xmin/xmax)/np.log(ymax15/ymax2)
            else:
                aspect_ratio = w/(h*5/6) if long else w/h
                datascaling  = np.log(xmin/xmax)/np.log(ymin/ymax)
            rotator = (datascaling/aspect_ratio)
            alpha = np.arctan(np.log(y1/y0)/np.log(x1/x0)*rotator)
            rotation =  np.rad2deg(alpha)#*rotator
            if y0>ymax and long:
                ax2.annotate(label, xy=(x0,y0), xytext=(0,5), textcoords="offset pixels",  rotation=rotation, rotation_mode='anchor', fontsize=fgtext, color=fg["color"], path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(1, -1)),], horizontalalignment="center")
            else:
                ax.annotate(label, xy=(x0,y0), xytext=(0,5), textcoords="offset pixels",  rotation=rotation, rotation_mode='anchor', fontsize=fgtext, color=fg["color"], path_effects=[path_effects.withSimplePatchShadow(alpha=0.8,offset=(1, -1)),], horizontalalignment="center")
    
    # ---- Data band ranges ----
    if long:
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
    bicep=False
    spider=False
    SPT=False
    act=False
    so = False
    dirbepos=1000 if long else 950
    spass_special = ymax2*yscaletext if pol else ymin*yscaletextup 
    databands = {"Haslam":  {"0.4\nHaslam": {"pol": False, "show": haslam, "position": [.408, ymin*yscaletextup],  "range": [.406,.410], "color": green,}},
                 "S-PASS":  {"2.3\nS-PASS":  {"pol": True, "show": spass,  "position": [2.35, spass_special],  "range": [2.1,2.4], "color": green,}},
                 "C-BASS":  {"5.0\nC-BASS":   {"pol": True, "show": cbass,  "position": [5., ymax2*yscaletext],    "range": [4.,6.], "color": blue,}},
                 "CHI-PASS":{"CHI-PASS\n1.4":{"pol": False, "show": chipass,"position": [1.35, ymin*yscaletextup],"range": [1.3945-0.064/2, 1.3945+0.064/2], "color": lightred,}},
                 "QUIJOTE": {"11":    {"pol": True, "show": quijote,"position": [10.5, ymax2*yscaletext],    "range":  [10.,12.], "color": red,},
                             "13":             {"pol": True, "show": quijote, "position": [13.5, ymax2*yscaletext], "range":  [12.,14.], "color": red,},
                             "17\nQUIJOTE":             {"pol": True, "show": quijote, "position": [17, ymax2*yscaletext], "range":  [16.,18.], "color": red,},
                             "19":             {"pol": True, "show": quijote, "position": [21, ymax2*yscaletext], "range":  [18.,21.], "color": red,},
                             ".\n31":             {"pol": True, "show": quijote, "position": [31, ymax2*yscaletext], "range":  [26.,36.], "color": red,},
                             ".\n41":             {"pol": True, "show": quijote, "position": [42, ymax2*yscaletext], "range":  [35.,47.], "color": red,}},
                 "Planck":  {"30":          {"pol": True, "show": planck, "position": [27,  ymax2*yscaletext], "range": [23.9,34.5],"color": orange,},      # Planck 30
                             "44":          {"pol": True, "show": planck, "position": [40,  ymax2*yscaletext], "range": [39,50]    ,"color": orange,},      # Planck 44
                             "70":          {"pol": True, "show": planck, "position": [60,  ymax2*yscaletext], "range": [60,78]    ,"color": orange,},      # Planck 70
                             "100": {"pol": True, "show": planck, "position": [90,  ymax2*yscaletext], "range": [82,120]   ,"color": orange,},      # Planck 100
                             "143\nPlanck":         {"pol": True, "show": planck, "position": [130, ymax2*yscaletext], "range": [125,170]  ,"color": orange,},      # Planck 143
                             "217":         {"pol": True, "show": planck, "position": [195, ymax2*yscaletext], "range": [180,265]  ,"color": orange,},      # Planck 217
                             "353":         {"pol": True, "show": planck, "position": [320, ymax2*yscaletext], "range": [300,430]  ,"color": orange,},      # Planck 353
                             "545":         {"pol": False, "show": planck, "position": [490, ymax2*yscaletext], "range": [450,650]  ,"color": orange,},      # Planck 545
                             "857":         {"pol": False, "show": planck, "position": [730, ymax2*yscaletext], "range": [700,1020] ,"color": orange,}},      # Planck 857
                 "DIRBE":   {"DIRBE\n1250":  {"pol": False, "show": dirbe, "position": [dirbepos, ymin*yscaletextup], "range": [1000,1540] , "color": red,},     # DIRBE 1250
                             "2140":         {"pol": False, "show": dirbe, "position": [1650, ymin*yscaletextup], "range": [1780,2500] , "color": red,},     # DIRBE 2140
                             "3000":         {"pol": False, "show": dirbe, "position": [2500, ymin*yscaletextup], "range": [2600,3500] , "color": red,}},     # DIRBE 3000
                 "WMAP":    {"K": {"pol": True, "show": wmap, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": teal,}, 
                             "WMAP\nKa":      {"pol": True, "show": wmap, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": teal,},
                             "Q":       {"pol": True, "show": wmap, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": teal,}, 
                             "V":       {"pol": True, "show": wmap, "position": [58.,  ymin*yscaletextup], "range": [54,68], "color": teal,}, 
                             "W":       {"pol": True, "show": wmap, "position": [90.,  ymin*yscaletextup], "range": [84,106], "color": teal,}}, 
                 "BICEPx/KECK":    {"95\nBICEPx/KECK": {"pol": True, "show": bicep, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": teal,}, 
                                    "150":      {"pol": True, "show": bicep, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": teal,},
                                    "220":       {"pol": True, "show": bicep, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": teal,}},
                 "SPT":    {"90": {"pol": True, "show": SPT, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": teal,}, 
                            "150":      {"pol": True, "show": SPT, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": teal,},
                            "220\nSPT":       {"pol": True, "show": SPT, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": teal,}}, 
                 "SPIDER":    {"100": {"pol": True, "show": spider, "position": [21.8, ymin*yscaletextup], "range": [21,25.5], "color": teal,}, 
                               "150":      {"pol": True, "show": spider, "position": [31.5, ymin*yscaletextup], "range": [30,37], "color": teal,},
                               "280\nSPIDER":       {"pol": True, "show": spider, "position": [39.,  ymin*yscaletextup], "range": [38,45], "color": teal,}}, 
                 "SO":     {"27":           {"pol": True, "show": so, "position": [27,  ymax2*yscaletext], "range": [23.9,34.5],"color": orange,},      # Planck 30
                            "39\nSO":       {"pol": True, "show": so, "position": [40,  ymax2*yscaletext], "range": [39,50]    ,"color": orange,},      # Planck 44
                             "93":          {"pol": True, "show": so, "position": [60,  ymax2*yscaletext], "range": [60,78]    ,"color": orange,},      # Planck 70
                             "145":         {"pol": True, "show": so, "position": [90,  ymax2*yscaletext], "range": [82,120]   ,"color": orange,},      # Planck 100
                             "225":         {"pol": True, "show": so, "position": [130, ymax2*yscaletext], "range": [125,170]  ,"color": orange,},      # Planck 143
                             "280":         {"pol": True, "show": so, "position": [195, ymax2*yscaletext], "range": [180,265]  ,"color": orange,}},      # Planck 217
                 "ACT":     {"28":          {"pol": True, "show": act, "position": [27,  ymax2*yscaletext], "range": [23.9,34.5],"color": orange,},      # Planck 30
                             "41":          {"pol": True, "show": act, "position": [40,  ymax2*yscaletext], "range": [39,50]    ,"color": orange,},      # Planck 44
                             "90\nACT":          {"pol": True, "show": act, "position": [60,  ymax2*yscaletext], "range": [60,78]    ,"color": orange,},      # Planck 70
                             "150": {"pol": True, "show": act, "position": [90,  ymax2*yscaletext], "range": [82,120]   ,"color": orange,},      # Planck 100
                             "220":         {"pol": True, "show": act, "position": [130, ymax2*yscaletext], "range": [125,170]  ,"color": orange,}},      # Planck 143            
    }
    
    bands2 = {"Planck": [30,44,70,100,143,216,353,545,857],
              "WMAP": [23,33,41,61,94],
              "S-PASS": [2.303],
              "C-BASS": [5.0],
              "CHI-PASS": [1.4],
              "QUIJOTE": [11, 13, 17, 19, 31, 41],
              "LiteBIRD": [40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402],
              "GroundBIRD": [145,220],
              "LSPE": [43,240],
              "QUBIC": [150, 220],
              "SO": [27,39,93,145,225,280],
              "ACT": [28,41,90, 150, 220,],
              "PICO": [21, 25, 30, 36, 43, 52, 62, 75, 90, 108, 129, 155, 186, 223, 268, 321, 385, 462, 555, 666, 799],
              "SPT": [90,150,220],
              "SPIDER": [100,150,280],
              "BICEP": [95,150,220],
              "CMB-S4": [30,40,85,95,145,155,220,270,20,27,39,93,145,225,278,27,39,93,145,225,278],
              "CMB-BHARAT": [60,600],
          }

    # Set databands from dictonary
    yrows=np.logspace(0,2,8)[::-1]
    cm = plt.cm.get_cmap('tab20')
    if False:
        c=0
        k=0
        for label, freqs in bands2.items():
            va="top"
            xpos=max(freqs)
            if xpos<10:
                ax2.text(xpos,ymax2*yscaletext, label, color=cm.colors[c], va=va, ha="center", size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
            else:
                if k>7:               
                    ax.text(4,yrows[k-8], label, color=cm.colors[c], va=va, ha="left", size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                else:
                    ax.text(1.4,yrows[k], label, color=cm.colors[c], va=va, ha="left", size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                k+=1

            for nu in freqs:
                ax.axvline(nu,color=cm.colors[c], alpha=0.4, zorder=0)
                if long:
                    ax2.axvline(nu,color=cm.colors[c],alpha=0.4,zorder=0)
            c+=1
    else:
        for experiment, bands in databands.items():
            for label, band in bands.items():
                if band["show"]:
                    #if label == "353" and not pol: continue # SPECIFIC FOR BP SETUP
                    if pol and not band["pol"]:
                        continue # Skip non-polarization bands
                    if band["position"][0]>=xmax or band["position"][0]<=xmin:
                        continue # Skip databands outside range

                    bottombands = ["WMAP", "CHI-PASS", "DIRBE", "Haslam"]
                    if not pol: 
                        bottombands.append("S-PASS")
                    va = "bottom" if experiment in bottombands else "top" # VA for WMAP on bottom
                    ha = "left" if experiment in ["Planck", "WMAP", "DIRBE",] else "center"
                    ax.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=-20, label=experiment)
                    if long:
                        ax2.axvspan(*band["range"], color=band["color"], alpha=0.3, zorder=-20, label=experiment)
                        if experiment in bottombands:
                            ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                        else:
                            ax2.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])
                    else:
                        ax.text(*band["position"], label, color=band["color"], va=va, ha=ha, size=freqtext, path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1,-1))])

    # ---- Axis stuff ----
    lsize=20
    """
    majorticks = minorticks = []
    majorticks_ = [1,10,100,1000]
    minorticks_ = [0.3,3,30,300,3000]
    for i, tick in enumerate(minorticks_):
        if tick>=xmin and tick<=xmax:
            minorticks.append(tick)
    for i, tick in enumerate(majorticks_):
        if tick>=xmin and tick<=xmax:
            majorticks.append(tick)
                        

    """

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
    if long:
        ax2.set(xscale='log', yscale='log', ylim=(ymax15, ymax2), xlim=(xmin,xmax), yticks=[1e4,1e6,], xticks=ticks, xticklabels=ticks)
        ax2.tick_params(axis='both', which='major', labelsize=lsize, direction='in')
        ax2.tick_params(which="both",direction="in")
        ax2.tick_params(axis='y', labelrotation=90,) 
        ax2.set_yticklabels([fmt(x,1) for x in ax2.get_yticks()], va="center")

    # Axis labels
    sax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        top=False,
        bottom=False,
        left=True,
        right=False,
        width=0.0,
    )
    if pol:
        sax.set_ylabel(r"RMS polarization amplitude [$\mu\mathrm{K}_{\mathrm{RJ}}$]",fontsize=lsize)
    else:
        sax.set_ylabel(r"RMS brightness temperature [$\mu\mathrm{K}_{\mathrm{RJ}}$]",fontsize=lsize)
    sax.set_xlabel(r"Frequency [GHz]",fontsize=lsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.02)

    #ax.legend(loc=6,prop={'size': 20}, frameon=False)

    # ---- Plotting ----
    #plt.tight_layout(h_pad=0.3)
    filename = "spectrum"
    filename += "_pol" if pol else ""
    filename += "_long" if long else ""
    filename += "_darkmode" if darkmode else ""
    filename += ".png" if png else ".pdf"
    print("Plotting {}".format(filename))
    plt.savefig(filename, bbox_inches='tight',  pad_inches=0.02, transparent=True)

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
        return r"$"+str(a)+"\cdot 10^{"+str(b)+"}$"

	
def gradient_fill(x, y, fill_color=None, ax=None, alpha=1.0, invert=False, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon

    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    #alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if invert:
        ymin,ymax = (ymax,ymin)
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def gradient_fill_between(ax, x, y1, y2, color="#ffa15a", alpha=0.8):
    N = 100
    y = np.zeros((N, len(y1)))
    for i in range(len(y1)):
        y[:,i] = np.linspace(y1[i], y2[i], N)

    alpha=np.linspace(0,alpha,100)**1.5
    for i in range(1,N):
        ax.fill_between(x, y[i-1], y[i], color=color, alpha=alpha[i], zorder=-10)
        ax.set_rasterization_zorder(-1)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


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
