import numpy as np
import healpy as hp
import astropy.units as u


@u.quantity_input(freq_ref=(None, u.Hz), fwhm=(None, u.arcmin, u.rad))
def to_stokes(input_map, unit=None, freq_ref=None, fwhm_ref=None, label=None):
    """Converts a Healpix-like map to a native Stokes map object. 
    
    A StokesMap has shape (3, nside). If the input map does not match this 
    shape, e.g, it is an intensity only map with shape (nside,) it will be 
    converted to a (3, nside) array with the Q and U stokes parameters set to 
    zeros.
    
    Args:
    -----
    input_map (np.ndarray, astropy.units.quantity.Quantity):
        A (3, nside) or (nside,) map containing stokes I or IQU parameters.
    unit (astropy.units.Unit):
        Units of the map object. If input_map already has units it will try to 
        convert its units to input_unit. Default: None
    freq_ref (astropy.units.quantity.Quantity):
        Reference frequencies of the input_map. If a single value is given, the 
        reference frequency will be assumed to be equal for all IQU parameters. 
        Default: None
    fwhm_ref (astropy.units.quantity.Quantity):
        The fwhm of the beam used in the input_map.
    label (str):
        A descriptive label for the map. Default: None

    """
    if freq_ref is not None:
        if freq_ref.ndim == 0:
            freq_ref = u.Quantity([freq_ref, freq_ref, freq_ref])

    if isinstance(input_map, np.ndarray):
        if input_map.ndim == 1:
            zeros = np.zeros_like(input_map)
            map_ = u.Quantity([input_map, zeros, zeros])
        else:
            map_ = u.Quantity(input_map)

        if isinstance(input_map, u.Quantity):
            if unit is not None: 
                map_.to(unit)

        return StokesMap(input_map=map_, 
                         freq_ref=freq_ref, 
                         fwhm_ref=fwhm_ref, 
                         label=label)

    if isinstance(input_map, StokesMap):
        map_ = input_map.data
        if unit is not None and unit != input_map.unit:
            map_.to(unit)
        if freq_ref is None:
            freq_ref = input_map.freq_ref
        if label is None:
            label = input_map.label
        if fwhm_ref is None:
            fwhm_ref = input_map.fwhm_ref
        return StokesMap(input_map=map_, 
                         freq_ref=freq_ref, 
                         fwhm_ref=fwhm_ref, 
                         label=label)

    else:
        raise NotImplementedError(
            f'type {type(input_map)} is not supported. Must be np.ndarray or '
            'astropy.units.quantity.Quantity'
        )



class StokesMap:
    """Stokes map object for IQU maps (custom array container). 
    
    The object provides a convenient interface for the common IQU map with 
    built in methods and attributes to access and evaluate commonly associated 
    properties of IQU maps.

    Args:
    -----
    input_map (astropy.units.quantity.Quantity):
        Healpix map of shape (3, nside) representing stokes IQU parameters.
    freq_ref (astropy.units.quantity.Quantity):
        List of reference frequencies of shape (3,) for each stokes parameter.
        If the map is unpolarized, i.e, Q and U is zeros, then freq_ref will 
        have the same value in all list elements for broadcasting purposes. 
        Default: None
    fwhm_ref (astropy.units.quantity.Quantity):
        The fwhm of the beam used for the input_map.
    label (str):
        A descriptive label for the map, e.g 'Dust'. Default: None

    """
    input_map : u.Quantity
    freq_ref : u.Quantity
    fwhm_ref : u.Quantity
    label : str

    @u.quantity_input(freq_ref=(None, u.Hz), fwhm_ref=(None, u.arcmin, u.rad))
    def __init__(self, input_map, freq_ref=None, fwhm_ref=None, label=None):
        self.data = input_map.astype(np.float32)
        self.freq_ref = freq_ref
        self.fwhm_ref = fwhm_ref
        self.label = label

        if not isinstance(input_map, u.Quantity):
            raise TypeError('input_map must be an astropy.Quantity object')
        if input_map.shape[0] != 3:
            raise ValueError('input_map shape must be (3, nside)')
        if self.freq_ref is not None:
            if self.freq_ref.shape != (3,):
                raise ValueError('freq_ref shape must be (3,)')
        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid')


    @property
    def I(self):
        """Returns the stokes I map"""
        return self.data[0]


    @property
    def Q(self):
        """Returns the stokes Q map"""
        return self.data[1]


    @property
    def U(self):
        """Returns the stokes U map"""
        return self.data[2]
        

    @property
    def P(self):
        """Polarized map signal. P = sqrt(Q^2 + U^2)"""
        if self._has_pol:
            return np.sqrt(self.Q**2 + self.U**2)

        return np.zeros_like(self.I)


    @property
    def unit(self):
        """Returns the units of the IQU map"""
        return self.data.unit


    @property
    def _has_pol(self):
        """Returns True if self.Q is non-zero. False otherwise"""
        if np.any(self.Q.value) and np.any(self.U.value):
            return True

        return False


    @property
    def nside(self):
        """Healpix map resolution"""
        return hp.npix2nside(len(self.I))


    @property
    def shape(self):
        """
        Returns the shape of the IQU map
        """
        return self.data.shape


    @property
    def ndim(self):
        """Returns the number of dimensions of the map"""
        return self.data.ndim


    def mask(self, mask, sigs=None):
        """Applies a mask to the data"""

        if sigs==None:
            sigs = ["I"]
            if self._has_pol:
                sigs += ["Q", "U"]

        for sig in sigs:
            m = getattr(self,sig)
            if not isinstance(m, np.ma.core.MaskedArray):
                m = hp.ma(m)
            m.mask = np.logical_not(mask)
            setattr(self, sig, m)


    def remove_md(self, mask, sig=None, remove_dipole=True, remove_monopole=True):
        """
        This function removes the mono and dipole of the signals in the map object.
        If you only wish to remove from 1 signal, pass [0,1,2]
        """
        if sig==None:
            sig = [0,1,2]
        pol = ["I", "Q", "U"][sig]
        data = getattr(self, pol)
        data = [data] if data.ndim == 1 else data
        for i, m in enumerate(data):
            # Make sure data is masked array type
            if not isinstance(m, np.ma.core.MaskedArray):
                m = hp.ma(m)
        
            if mask == "auto":
                mono, dip = hp.fit_dipole(m, gal_cut=30)
            else:
                m_masked = m
                m_masked.mask = np.logical_not(mask)
                mono, dip = hp.fit_dipole(m_masked)

            # Subtract dipole map from data
            if isinstance(remove_dipole, np.ndarray):
                print("Removing dipole:")
                print(f"Dipole vector: {dip}")
                print(f"Dipole amplitude: {np.sqrt(np.sum(dip ** 2))}")

                # Create dipole template
                ray = range(len(m))
                vecs = hp.pix2vec(self.nside, ray)
                dipole = np.dot(dip, vecs)
                
                m = m - dipole

            if isinstance(remove_monopole, np.ndarray):
                print(f"Removing monopole:")
                print(f"Mono: {mono}")
                m = m - mono
            
            setattr(self, pol[i], m)


    def to_nside(self, new_nside):
        """ud_grades all stokes parameters to a new nside.

        Args:
        -----
        new_nside (int):
            Healpix map resolution parameter.

        """
        if not hp.isnsideok(new_nside, nest=True):
            raise ValueError(f'nside: {new_nside} is not valid.')
        if new_nside == self.nside:
            return

        self.data = hp.ud_grade(self.data.value, new_nside)*self.data.unit


    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad))
    def to_fwhm(self, fwhm):
        """Smooths the I, Q and U maps to a given fwhm.
        
        Args:
        -----
        fwhm (astropy.units.quantity.Quantity):
            The fwhm of the Gaussian used to smooth the map. Must be either in
            units of arcmin, degrees or radians.

        """
        fwhm = fwhm.to(u.rad)
        if self.fwhm_ref != fwhm:
            diff_fwhm = np.sqrt(fwhm.value**2 - self.fwhm_ref.value**2)
            if diff_fwhm < 0:
                raise ValueError(
                    'cannot smooth to a higher resolution '
                    f'(map fwhm: {self.fwhm_ref}).'
                )
            self.data = hp.smoothing(self.data, diff_fwhm)*self.data.unit
            self.fwhm_ref = fwhm


    def __array__(self):
        return np.array(self.data.value)


    def __add__(self, other):        
        return self.__class__(self.data + other)

    
    def __sub__(self, other):
        return self.__class__(self.data - other)


    def __mul__(self, other):
        return self.__class__(self.data * other)


    def __truediv__(self, other):
        return self.__class__(self.data / other)


    def __pow__(self, other):
        return self.__class__(self.data ** other)


    # Operations commute
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__


    def __iter__(self):
        return iter(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


    def __repr__(self):
        return f'{self.data}'