import astropy.units as u
import h5py
import healpy as hp
import numpy as np
import numba
import os
import pathlib
import sys

class Chain:
    """
    Commander3 chainfile object.

    """

    def __init__(self, h5file):
        """
        Initializes the chain object from a give Commander3 chainfile.

        Parameters
        ----------
        h5file : str, 'pathlib.Path'
            Commander3 chainfile. Must be a hdf5 file.

        """

        self.data = self.get_data(h5file)
        self.components = self.get_components()
        self.model_params = self.get_model_params()


    @staticmethod
    def get_data(h5file):
        """
        Validates that the given file is of a type and format that matches a
        Commander3 hdf5 output chainfile, and then returns file as a pathlib
        object.

        Parameters
        ----------
        filename_or_dir : str, 'pathlib.Path'
            Path to a Commander HDF5 datafile or a Commander chain directory
            containing a HDF5 datafile.

        Returns
        -------
        pathlib.Path
            pathlib Path object to Commander HDF5 chain file.

        """
        try:
            path = pathlib.Path(h5file)
        except:
            raise TypeError(
                f'Data input to Cosmoglobe() must be str or a os.PathLike '
                'object.'
            ) 

        if os.path.isfile(h5file):
            if path.name.endswith('.h5'):
                try: 
                    with h5py.File(h5file,'r') as f:
                        pass
                except OSError:
                    raise(OSError(
                        f'{path.name} has an invalid hdf5 format. Make sure '
                        'the file was created with a recent version of '
                        'Commander.'
                    ))
                return path

            else:
                raise OSError("Input file must be an hdf5 file.")

        elif os.path.isdir(h5file):
            chainfiles = []
            for file in os.listdir(h5file):
                if file.endswith('.h5'):
                    chainfiles.append(file)

            if len(chainfiles) > 1:
                print(f'Data directory contains more than one hdf5 file: '
                      f'{chainfiles} \nPlease provide a path including '
                      'the specific hdf5 file you wish to use.'
                )
                sys.exit()
            else:
                raise FileNotFoundError(
                    f"'{path.name}' directory does not contain a valid "
                    "chain HDF5 file."
                )

        else:
            raise FileNotFoundError(
                f"Could not find file or directory '{path.name}'. " 
            )
      
      
    def get_components(self):
        """
        Returns all sky components available in the commander output.

        Parameters
        ----------

        Returns
        -------
        component_list : list of strings
            List containing all avaiable sky components for a given chainsfile.

        """
        component_list = []
        with h5py.File(self.data,'r') as f:
            samples = sorted(list(f.keys()))
            components = f[(f'{samples[0]}')]

            for component in components:
                component_list.append(component)

            if not component_list:
                raise ValueError(
                    f'"{self.data.name}" doest not include any components.'
            )

        return component_list


    def get_model_params(self):
        """
        Returns a dictionary containing all model parameters from the commander 
        output.

        Parameters
        ----------

        Returns
        -------
        model_params : dict
            Dictionary containing commander model parameters.

        """   
        model_params = {}
        with h5py.File(self.data,'r') as f:
            samples = sorted(list(f.keys()))
            if samples[-1] == 'parameters':
                models = f[(f'{samples[-1]}')]
            else:
                raise KeyError(
                    f'Chainfile does not contain any model parameters. Make ' 
                    'sure the chainfile was produced with an updated '
                    'version of Commander3.'
            )
            for model in models:
                model_params[model] = {}

                params = models[model]
                for param in params:
                    if isinstance(models[model][param][()], bytes):
                        model_params[model][param] = models[model][param][()].decode("utf-8")

                    else:
                        if len(np.shape(models[model][param][()])) > 0:
                            model_params[model][param] = models[model][param][()]
                        else:
                            model_params[model][param] = models[model][param][()].item()


        for model in model_params:
            model_params[model]['fwhm'] *= u.arcmin
            model_params[model]['nu_ref'] *= u.GHz
            model_params[model]['nu_ref'] *= 1e-9

            if model_params[model]['polarization'] == 'True':
                model_params[model]['polarization'] = True
            elif model_params[model]['polarization'] == 'False':
                model_params[model]['polarization'] = False

        return model_params


    def get_item(self, component, item , sample='000000'):
        """
        Returns a specific item from the data.

        Parameters
        ----------
        component : str
            Name of component that contains item.
        item : str
            Name of item to be extracted from datafile.
        sample : str
            Sample number. Default is 000000.

        Returns
        -------
        item : int, float, str, numpy.ndarray
            item extracted from data. Can be any type.

        """  
        with h5py.File(self.data,'r') as f:
            samples = sorted(list(f.keys()))
            if sample in samples:
                models = f[(f'{sample}')]
            else:
                raise KeyError(
                    f"Chainfile does not contain sample '{sample}'."
            )

            components = sorted(list(models.keys()))
            if component in components:
                quantities = models[(f'{component}')]
            else:
                raise KeyError(
                    f'Chainfile does not contain data for component: '
                    '{component}.'
            )
            
            if item in quantities:
                item = quantities[(f'{item}')]
                if len(np.shape(item)) > 0:
                    if np.shape(item[()])[0] == 1:
                        return item[()][0]
                    else:
                        return item[()]
                else:
                    if np.shape(item[()])[0] == 1:
                        return item[()].item()[0]
                    else:
                        return item[()].item()


    def get_alm_list(self, component):
        """
        Returns a list of all available alms for a component in the commander 
        output.

        Parameters
        ----------
        component : str
            Component name.
        
        Returns
        -------
        alm_list : list of strings
            List containing all avaiable alms for a given component.

        """
        alm_list = []
        with h5py.File(self.data,'r') as f:
            samples = sorted(list(f.keys()))
            try:
                params = f[(f'{samples[0]}')][component]
            except:
                raise KeyError(f'"{component}" is not a valid component.')

            for param in params:
                if 'alm' in param:
                    alm_param = param.split('_')[0]
                    alm_list.append(alm_param)
            
            if not alm_list:
                raise ValueError(f'"{component}" doest not include any alms.')

        return alm_list


    def get_alms(self, alm_param, component, nside, polarization, fwhm,
                 multipole=None):
        """
        Returns array or arrays containing alms from Commander chain HDF5 file.

        Parameters
        ----------
        alm_param : str, optional
            alm parameter ('amp', 'T', 'beta') to extract from hdf5 file.
            Default is 'amp'.
        component : str
            component name.
        nside: int
            Healpix map resolution.
        lmax: int, optional
            Maximum value for l used in data.
            Default is None.
        fwhm : float, scalar, optional
            The fwhm of the Gaussian used to smooth the map (applied on alm)
            [in degrees]
            Default is 0.0.
        multipole : int
            A specific multipole. If provided, get_alms will extract
            only this specific multipole. Defaults to None.

        Returns
        -------
        alm_map : 'numpy.ndarray' or list of 'numpy.ndarray'
            A Healpix map in RING scheme at nside or a list of I,Q,U maps (if
            polarized input)

        """
        with h5py.File(self.data,'r') as f:
            samples = sorted(list(f.keys()))

            try:
                data = f[(f'{samples[0]}')][component]
            except:
                raise KeyError(f'"{component}" is not a valid component.')

            try:
                alms = data[f'{alm_param}_alm'][()]

            except:
                raise KeyError(f'"{alm_param}_alm" does not exist in file.')
            
            try:
                lmax = data[f'{alm_param}_lmax'][()]
            except:
                raise KeyError(f'"{alm_param}_lmax" does not exist in file.')

            if lmax <= 1:
                polarization = False

            if multipole is not None:
                alms_unpacked = unpack_multipole(alms, lmax, multipole)

            else:
                alms_unpacked = unpack_alms(alms, lmax)


            if hp.isnsideok(nside, nest=True):
                alms_map = hp.alm2map(alms_unpacked, 
                                      nside=nside, 
                                      lmax=lmax, 
                                      mmax=lmax, 
                                      fwhm=fwhm.to('rad').value, 
                                      pol=polarization, pixwin=True)
            else:
                print(f'nside: {nside} is not valid.')
                sys.exit()

        return alms_map




#Fix to OMP: Error #15 using numba
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@numba.njit(cache=True, fastmath=True)
def unpack_multipole(data, lmax, multipole=None):
    """
    Unpacks a single multipole alm from the Commander chain output.

    Parameters
    ----------
    data : HDF5 data
        Chain output hdf5 data.
    lmax : int
        Maximum value for l used in the data.

    Returns
    -------
    alms : 'numpy.ndarray'
        Unpacked version of the Commander alms (2-dimensional array)

    Unpacking algorithm: 
    "https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1
    262/src/tools.py#L9"

    """
    n = len(data)
    n_alms = int(lmax * ((2*lmax + 1 - lmax)/2) + lmax + 1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax+1):
            if l is multipole:
                j_real = l**2 + l
                alms[sigma, i] = complex(data[sigma, j_real], 0.0)
            i += 1
        
        for m in range(1, lmax+1):
            for l in range(m, lmax+1):
                if l is multipole:
                    j_real = l**2 + l + m
                    j_comp = l**2 + l - m
                    alms[sigma, i] = (complex(data[sigma, j_real], 
                                             data[sigma, j_comp],)
                                      /np.sqrt(2.0))
                i += 1
    return alms

@numba.njit(cache=True, fastmath=True)
def unpack_alms(data, lmax):
    """
    Unpacks alms from the Commander chain output.

    Parameters
    ----------
    data : HDF5 data
        Chain output hdf5 data.
    lmax : int
        Maximum value for l used in the data.

    Returns
    -------
    alms : 'numpy.ndarray'
        Unpacked version of the Commander alms (2-dimensional array)

    Unpacking algorithm: 
    "https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1
    262/src/tools.py#L9"

    """
    n = len(data)
    n_alms = int(lmax * (2*lmax+1 - lmax) / 2 + lmax+1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax+1):
            j_real = l**2 + l
            alms[sigma, i] = complex(data[sigma, j_real], 0.0)
            i += 1

        for m in range(1, lmax+1):
            for l in range(m, lmax+1):
                j_real = l**2 + l + m
                j_comp = l**2 + l - m
                alms[sigma, i] = complex(data[sigma, j_real], 
                                         data[sigma, j_comp],)/ np.sqrt(2.0)
                i += 1

    return alms