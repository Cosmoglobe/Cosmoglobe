import numba
import numpy as np
#######################
# HELPFUL TOOLS BELOW #
#######################


@numba.njit(cache=True, fastmath=True)  # Speeding up by a lot!
def unpack_alms(maps, lmax):
    #print("Unpacking alms")
    mmax = lmax
    nmaps = len(maps)
    # Nalms is length of target alms
    Nalms = int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)
    alms = np.zeros((nmaps, Nalms), dtype=np.complex128)
    # Unpack alms as output by commander
    for sig in range(nmaps):
        i = 0
        for l in range(lmax+1):
            j_real = l ** 2 + l
            alms[sig, i] = complex(maps[sig, j_real], 0.0)
            i += 1

        for m in range(1, lmax + 1):
            for l in range(m, lmax + 1):
                j_real = l ** 2 + l + m
                j_comp = l ** 2 + l - m

                alms[sig, i] = complex(maps[sig, j_real], maps[sig, j_comp],) / np.sqrt(2.0)

                i += 1
    return alms

def alm2fits_tool(input, dataset, nside, lmax, fwhm, save=True,):
    """
    Function for converting alms in hdf file to fits
    """
    import h5py
    import healpy as hp

    try:
        sample = int(dataset.split("/")[0])
        print(f"Using sample {sample}")
    except:
        print(f"No sample specified, fetching last smample")
        with h5py.File(input, "r") as f:
            sample = str(len(f.keys()) - 2).zfill(6)
            dataset=sample+"/"+dataset
        print(f"Sample {sample} found, dataset now {dataset}")

    with h5py.File(input, "r") as f:
        alms = f[dataset][()]
        lmax_h5 = f[f"{dataset[:-3]}lmax"][()]  # Get lmax from h5

    if lmax:
        # Check if chosen lmax is compatible with data
        if lmax > lmax_h5:
            print(
                "lmax larger than data allows: ", lmax_h5,
            )
            print("Please chose a value smaller than this")
    else:
        # Set lmax to default value
        lmax = lmax_h5
    mmax = lmax

    alms_unpacked = unpack_alms(alms, lmax)  # Unpack alms
    # If not amp map, set spin 0.
    if "amp_alm" in dataset:
        pol = True
        if np.shape(alms_unpacked)[0]==1:
            pol = False
    else:
        pol = False
    

    print(f"Making map from alms, setting lmax={lmax}, pol={pol}")
    maps = hp.sphtfunc.alm2map(alms_unpacked, int(nside), lmax=int(lmax), mmax=int(mmax), fwhm=arcmin2rad(fwhm), pol=pol, pixwin=True,)
    outfile = dataset.replace("/", "_")
    outfile = outfile.replace("_alm", "")
    if save:
        outfile += f"_{str(int(fwhm))}arcmin" if fwhm > 0.0 else ""
        hp.write_map(outfile + f"_n{str(nside)}_lmax{lmax}.fits", maps, overwrite=True, dtype=None)
    return maps, nside, lmax, fwhm, outfile

def h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, command, pixweight=None, zerospin=False, lowmem=False, notchain=False):
    """
    Function for calculating mean and stddev of signals in hdf file
    """
    # Check if you want to output a map
    import h5py
    import healpy as hp
    from tqdm import tqdm

    if (lowmem and command == np.std): #need to compute mean first
        mean_data = h5handler(input, dataset, min, max, maxchain, output, fwhm, nside, np.mean, pixweight, zerospin, lowmem,)

    print()
    if command: print("{:-^50}".format(f" {dataset} calculating {command.__name__} "))
    print("{:-^50}".format(f" nside {nside}, {fwhm} arcmin smoothing "))

    if dataset.endswith("map"):
        type = "map"
    elif dataset.endswith("rms"):
        type = "map"
    elif dataset.endswith("alm"):
        type = "alm"
    elif dataset.endswith("sigma"):
        type = "sigma"
    else:
        type = "data"

    if (lowmem):
        nsamp = 0 #track number of samples
        first_samp = True #flag for first sample
    else:
        dats = []

    use_pixweights = False if pixweight == None else True
    maxnone = True if max == None else False  # set length of keys for maxchains>1
    pol = True if zerospin == False else False  # treat maps as TQU maps (polarization)
    for c in range(1, maxchain + 1):
        filename = input.replace("c0001", "c" + str(c).zfill(4))
        with h5py.File(filename, "r") as f:

            if notchain:
                data = f[dataset][()]
                if data.shape[0] == 1:
                    # Make sure its interprated as I by healpy
                    # For non-polarization data, (1,npix) is not accepted by healpy
                    data = data.ravel()
                dats.append(data)
                continue

            if maxnone:
                # If no max is specified, chose last sample
                max = len(f.keys()) - 2

            print("{:-^48}".format(f" Samples {min} to {max} in {filename}"))

            for sample in tqdm(range(min, max + 1), ncols=80):
                # Identify dataset
                # alm, map or (sigma_l, which is recognized as l)

                # Unless output is ".fits" or "map", don't convert alms to map.
                alm2map = True if output.endswith((".fits", "map")) else False

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
                        tag = f"{tag[:-3]}map"
                        print(f"WARNING! No {type} data found, switching to map.")
                        data = f[tag][()]
                        type = "map"
                except:
                    print(f"Found no dataset called {dataset}")
                    print(f"Trying alms instead {tag}")
                    try:
                        # Use alms instead (This takes longer and is not preferred)
                        tag = f"{tag[:-3]}alm"
                        type = "alm"
                        data = f[tag][()]
                    except:
                        print("Dataset not found.")

                # If data is alm, unpack.
                if type == "alm":
                    lmax_h5 = f[f"{tag[:-3]}lmax"][()]
                    data = unpack_alms(data, lmax_h5)  # Unpack alms
                    
                if data.shape[0] == 1:
                    # Make sure its interprated as I by healpy
                    # For non-polarization data, (1,npix) is not accepted by healpy
                    data = data.ravel()

                # If data is alm and calculating std. Bin to map and smooth first.
                if type == "alm" and command == np.std and alm2map:
                    #print(f"#{sample} --- alm2map with {fwhm} arcmin, lmax {lmax_h5} ---")
                    data = hp.alm2map(data, nside=nside, lmax=lmax_h5, fwhm=arcmin2rad(fwhm), pixwin=True,pol=pol,)

                # If data is map, smooth first.
                elif type == "map" and fwhm > 0.0 and (command == np.std or command == np.cov):
                    #print(f"#{sample} --- Smoothing map ---")
                    if use_pixweights:
                        data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm),pol=pol,use_pixel_weights=True,datapath=pixweight)
                    else: #use ring weights
                        data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm),pol=pol,use_weights=True)

                if (lowmem):
                    if (first_samp):
                        first_samp=False
                        if (command==np.mean):
                            dats=data.copy()
                        elif (command==np.std):
                            dats=(mean_data - data)**2
                        else:
                            print('     Unknown command {command}. Exiting')
                            exit()
                    else:
                        if (command==np.mean):
                            dats=dats+data
                        elif (command==np.std):
                            dats=dats+(mean_data - data)**2
                    nsamp+=1
                else:
                    # Append sample to list
                    dats.append(data)

    if (lowmem):
        if (command == np.mean):
            outdata = dats/nsamp
        elif (command == np.std):
            outdata = np.sqrt(dats/nsamp)
    else:
        # Convert list to array
        dats = np.array(dats)
        # Calculate std or mean
        print(dats.shape)
        if (command == np.cov):
            N  = dats.shape[0]
            m1 = dats - dats.sum(axis=0, keepdims=1)/N
            outdata = np.einsum('ijk,imk->jmk', m1, m1)/N
        else:
            outdata = command(dats, axis=0)
            outdata = command(dats, axis=0) if command else dats
    # Smoothing afterwards when calculating mean
    if type == "alm" and command == np.mean and alm2map:
        print(f"# --- alm2map mean with {fwhm} arcmin, lmax {lmax_h5} ---")
        outdata = hp.alm2map(
            outdata, nside=nside, lmax=lmax_h5, fwhm=arcmin2rad(fwhm), pixwin=True, pol=pol
        )

    if type == "map" and fwhm > 0.0 and command == np.mean:
        print(f"--- Smoothing mean map with {fwhm} arcmin,---")
        if use_pixweights:
            outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm),pol=pol,use_pixel_weights=True,datapath=pixweight)
        else: #use ring weights
            outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm),pol=pol,use_weights=True)

    # Outputs fits map if output name is .fits
    if output.endswith(".fits"):
        hp.write_map(output, outdata, overwrite=True, dtype=None)
    elif output.endswith(".dat"):
        while np.ndim(outdata)>2: 
            if outdata.shape[-1]==4:
                tdata = outdata[:,0,0]
                print(tdata)
                outdata = outdata[:,:,3]
                outdata[:,0] = tdata
            else:
                outdata = outdata[:,:,0]

        np.savetxt(output, outdata)
    return outdata

def arcmin2rad(arcmin):
    return arcmin * (2 * np.pi) / 21600

def legend_positions(df, y, scaling):
    """
    Calculate position of labels to the right in plot... 
    """
    positions = {}
    for column in y:
        positions[column] = df[column].values[-1] - 0.005

    def push(dpush):
        """
        ...by puting them to the last y value and
        pushing until no overlap
        """
        collisions = 0
        for column1, value1 in positions.items():
            for column2, value2 in positions.items():
                if column1 != column2:
                    dist = abs(value1-value2)
                    if dist < scaling:# 0.075: #0.075: #0.023:
                        collisions += 1
                        if value1 < value2:
                            positions[column1] -= dpush
                            positions[column2] += dpush
                        else:
                            positions[column1] += dpush
                            positions[column2] -= dpush
                            return True
    dpush = .001
    pushings = 0
    while True:
        if pushings == 1000:
            dpush*=10
            pushings = 0
        pushed = push(dpush)
        if not pushed:
            break

        pushings+=1

    return positions

def cmb(nu, A):
    """
    CMB blackbody spectrum
    """
    h = 6.62607e-34 # Planck's konstant
    k_b  = 1.38065e-23 # Boltzmanns konstant
    Tcmb = 2.7255      # K CMB Temperature

    x = h*nu/(k_b*Tcmb)
    g = (np.exp(x)-1)**2/(x**2*np.exp(x))
    s_cmb = A/g
    return s_cmb

def sync(nu, As, alpha, nuref=0.408):
    """
    Synchrotron spectrum using template
    """
    print("nuref", nuref)
    #alpha = 1., As = 30 K (30*1e6 muK)
    nu_0 = nuref*1e9 # 408 MHz
    from pathlib import Path
    synch_template = Path(__file__).parent / "Synchrotron_template_GHz_extended.txt"
    fnu, f = np.loadtxt(synch_template, unpack=True)
    f = np.interp(nu, fnu*1e9, f)
    f0 = np.interp(nu_0, nu, f) # Value of s at nu_0
    s_s = As*(nu_0/nu)**2*f/f0
    return s_s

def ffEM(nu,EM,Te):
    """
    Freefree spectrum using emission measure
    """
    #EM = 1 cm-3pc, Te= 500 #K
    T4 = Te*1e-4
    nu9 = nu/1e9 #Hz
    g_ff = np.log(np.exp(5.960-np.sqrt(3)/np.pi*np.log(nu9*T4**(-3./2.)))+np.e)
    tau = 0.05468*Te**(-3./2.)*nu9**(-2)*EM*g_ff
    s_ff = 1e6*Te*(1-np.exp(-tau))
    return s_ff

def ff(nu,A,Te, nuref=40.):
    """
    Freefree spectrum
    """
    h = 6.62607e-34 # Planck's konstant
    k_b  = 1.38065e-23 # Boltzmanns konstant

    nu_ref = nuref*1e9
    S =     np.log(np.exp(5.960 - np.sqrt(3.0)/np.pi * np.log(    nu/1e9*(Te/1e4)**-1.5))+2.71828)
    S_ref = np.log(np.exp(5.960 - np.sqrt(3.0)/np.pi * np.log(nu_ref/1e9*(Te/1e4)**-1.5))+2.71828)
    s_ff = A*S/S_ref*np.exp(-h*(nu-nu_ref)/k_b/Te)*(nu/nu_ref)**-2
    return s_ff

def sdust(nu, Asd, nu_p, polfrac, fnu = None, f_ = None, nuref=22.,):
    """
    Spinning dust spectrum using spdust2
    """
    nuref = nuref*1e9 
    scale = 30./nu_p

    try:
        f = np.interp(scale*nu, fnu, f_)
        f0 = np.interp(scale*nuref, fnu, f_) # Value of s at nu_0
    except:
        from pathlib import Path
        ame_template = Path(__file__).parent / "spdust2_cnm.dat"
        fnu, f_ = np.loadtxt(ame_template, unpack=True)
        fnu *= 1e9
        f = np.interp(scale*nu, fnu, f_)
        f0 = np.interp(scale*nuref, fnu, f_) # Value of s at nu_0
        
    s_sd = polfrac*Asd*(nuref/nu)**2*f/f0
    return s_sd

def tdust(nu,Ad,betad,Td,nuref=545.):
    """
    Thermal dust modified blackbody spectrum.
    """
    h = 6.62607e-34 # Planck's konstant
    k_b  = 1.38065e-23 # Boltzmanns konstant
    nu0=nuref*1e9
    gamma = h/(k_b*Td)
    s_d=Ad*(nu/nu0)**(betad+1)*(np.exp(gamma*nu0)-1)/(np.exp(gamma*nu)-1)
    return s_d

def lf(nu,Alf,betalf,nuref=30e9):
    """
    low frequency component spectrum (power law)
    """
    return Alf*(nu/nuref)**(betalf)

def line(nu, A, freq, conversion=1.0):
    """
    Line emission spectrum
    """
    if isinstance(nu, np.ndarray):
        return np.where(np.isclose(nu, 1e9*freq), A*conversion, 0.0)
    else:
        if np.isclose(nu, 1e9*freq[0]):
            return A*conversion
        else:
            return 0.0

def rspectrum(nu, r, sig, scaling=1.0):
    """
    Calculates the CMB amplituded given a value of r and requested modes
    """
    import camb
    from camb import model, initialpower
    import healpy as hp
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=r)
    lmax=6000
    pars.set_for_lmax(lmax,  lens_potential_accuracy=0)
    pars.WantTensors = True
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(params=pars, lmax=lmax, CMB_unit='muK', raw_cl=True,)


    l = np.arange(2,lmax+1)

    if sig == "TT":
        cl = powers['unlensed_scalar']
        signal = 0
    elif sig == "EE":
        cl = powers['unlensed_scalar']
        signal = 1
    elif sig == "BB":
        cl = powers['tensor']
        signal = 2

    bl = hp.gauss_beam(40/(180/np.pi*60), lmax,pol=True)
    A = np.sqrt(sum( 4*np.pi * cl[2:,signal]*bl[2:,signal]**2/(2*l+1) ))
    return cmb(nu, A*scaling)

def fits_handler(input, min, max, minchain, maxchain, chdir, output, fwhm, nside, zerospin, drop_missing, pixweight, command, lowmem=False, fields=None, write=False):
    """
    Function for handling fits files.
    """
    # Check if you want to output a map
    import healpy as hp
    from tqdm import tqdm
    import os

    if maxchain is None:
        maxchain = minchain + 1

    if (not input.endswith(".fits")):
        print("Input file must be a '.fits'-file")
        exit()

    if (lowmem and command == np.std): #need to compute mean first
        mean_data = fits_handler(input, min, max, minchain, maxchain, chdir, output, fwhm, nside, zerospin, drop_missing, pixweight, lowmem, np.mean, fields, write=False)

    if (minchain > maxchain):
        print('Minimum chain number larger that maximum chain number. Exiting')
        exit()
    aline=input.split('/')
    dataset=aline[-1]
    print()
    if command: print("{:-^50}".format(f" {dataset} calculating {command.__name__} "))
    if (nside == None):
        print("{:-^50}".format(f" {fwhm} arcmin smoothing "))
    else:
        print("{:-^50}".format(f" nside {nside}, {fwhm} arcmin smoothing "))

    type = 'map'

    if (not lowmem):
        dats = []

    nsamp = 0 #track number of samples
    first_samp = True #flag for first sample

    use_pixweights = False if pixweight == None else True
    maxnone = True if max == None else False  # set length of keys for maxchains>1
    pol = True if zerospin == False else False  # treat maps as TQU maps (polarization)
    for c in range(minchain, maxchain + 1):
        if (chdir==None):
            filename = input.replace("c0001", "c" + str(c).zfill(4))
        else:
            if maxchain > minchain + 1:
                filename = chdir+'_c%i/'%(c)+input
            else:
                filename = f'{chdir}/{input}'
        basefile = filename.split("k000001")

        if maxnone:
            # If no max is specified, find last sample of chain
            # Assume residual file of convention res_label_c0001_k000234.fits, 
            # i.e. final numbers of file are sample number
            max_found = False
            siter=min
            while (not max_found):
                filename = basefile[0]+'k'+str(siter).zfill(6)+basefile[1]

                if (os.path.isfile(filename)):
                    siter += 1
                else:
                    max_found = True
                    max = siter - 1

        else:
            if (first_samp):
                for chiter in range(minchain,maxchain + 1):
                    if (chdir==None):
                        tempname = input.replace("c0001", "c" + str(c).zfill(4))
                    else:
                        tempname = chdir+'_c%i/'%(c)+input
                        temp = tempname.split("k000001")

                    for siter in range(min,max+1):
                        tempf = temp[0]+'k'+str(siter).zfill(6)+temp[1]

                        if (not os.path.isfile(tempf)):
                            print('chain %i, sample %i missing'%(c,siter))
                            print(tempf)
                            if (not drop_missing):
                                exit()


        print("{:-^48}".format(f" Samples {min} to {max} in {filename}"))

        for sample in tqdm(range(min, max + 1), ncols=80):
                # dataset sample formatting
                filename = basefile[0]+'k'+str(sample).zfill(6)+basefile[1]                
                if (first_samp):
                    # Check which fields the input maps have
                    if (not os.path.isfile(filename)):
                        if (not drop_missing):
                            exit()
                        else:
                            continue
                    
                    _, header = hp.fitsfunc.read_map(filename, h=True, dtype=None)
                    if fields!=None:
                        nfields = 0
                        for par in header:
                            if (par[0] == 'TFIELDS'):
                                nfields = par[1]
                                break
                        if (nfields == 0):
                            print('No fields/maps in input file')
                            exit()
                        elif (nfields == 1):
                            fields=(0)
                        elif (nfields == 2):
                            fields=(0,1)
                        elif (nfields == 3):
                            fields=(0,1,2)
                    #print('   Reading fields ',fields)

                    nest = False
                    for par in header:
                        if (par[0] == 'ORDERING'):
                            if (not par[1] == 'RING'):
                                nest = True
                            break

                    nest = False
                    for par in header:
                        if (par[0] == 'NSIDE'):
                            nside_map = par[1]
                            break


                    if (not nside == None):
                        if (nside > nside_map):
                            print('   Specified nside larger than that of the input maps')
                            print('   Not up-grading the maps')
                            print('')

                if (not os.path.isfile(filename)):
                    if (not drop_missing):
                        exit()
                    else:
                        continue

                data = hp.fitsfunc.read_map(filename,field=fields,h=False, nest=nest, dtype=None)
                if (nest): #need to reorder to ring-ordering
                    data = hp.pixelfunc.reorder(data,n2r=True)

                # degrading if relevant
                if (not nside == None):
                    if (nside < nside_map):
                        data=hp.pixelfunc.ud_grade(data,nside) #ordering=ring by default

                if data.shape[0] == 1:
                    # Make sure its interprated as I by healpy
                    # For non-polarization data, (1,npix) is not accepted by healpy
                    data = data.ravel()

                # If smoothing applied and calculating stddev, smooth first.
                if fwhm > 0.0 and command == np.std:
                    #print(f"#{sample} --- Smoothing map ---")
                    if use_pixweights:
                        data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm),pol=pol,use_pixel_weights=True,datapath=pixweight)
                    else: #use ring weights
                        data = hp.sphtfunc.smoothing(data, fwhm=arcmin2rad(fwhm),pol=pol,use_weights=True)
                    
                if (lowmem):
                    if (first_samp):
                        if (command==np.mean):
                            dats=data.copy()
                        elif (command==np.std):
                            dats=(mean_data - data)**2
                        else:
                            print('     Unknown command {command}. Exiting')
                            exit()
                    else:
                        if (command==np.mean):
                            dats=dats+data
                        elif (command==np.std):
                            dats=dats+(mean_data - data)**2
                    nsamp+=1
                else:
                    # Append sample to list
                    dats.append(data)
                first_samp=False

    if (lowmem):
        if (command == np.mean):
            outdata = dats/nsamp
        elif (command == np.std):
            outdata = np.sqrt(dats/nsamp)
    else:
        # Convert list to array
        dats = np.array(dats)
        # Calculate std or mean
        if (command == np.cov):
            N  = dats.shape[0]
            m1 = dats - dats.sum(axis=0, keepdims=1)/N
            outdata = np.einsum('ijk,imk->jmk', m1, m1)/N
        else:
            outdata = command(dats, axis=0)

    # Smoothing afterwards when calculating mean
    if fwhm > 0.0 and command == np.mean:
        print(f"--- Smoothing mean map with {fwhm} arcmin,---")
        if use_pixweights:
            outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm),pol=pol,use_pixel_weights=True,datapath=pixweight)
        else: #use ring weights
            outdata = hp.sphtfunc.smoothing(outdata, fwhm=arcmin2rad(fwhm),pol=pol,use_weights=True)
    # Outputs fits map if output name is .fits
    if write:
        print("debug 3")
        if output.endswith(".fits"):
            hp.write_map(output, outdata, overwrite=True, dtype=None)
        elif output.endswith(".dat"):
            np.savetxt(output, outdata)
    else:
        return outdata

