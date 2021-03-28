import numpy as np
import healpy as hp
import astropy.units as u
import operator
import sys


@u.quantity_input(nu_ref=(None, u.Hz))
def to_IQU(input_map, unit=None, nu_ref=None, label=None):
    """Converts a map to a native IQU map object.
    
    Args:
    -----
    input_map : np.ndarray, astropy.units.quantity.Quantity
        Map to be converted.
    unit : astropy.units.Unit
        Units of the map object. If input_map already has units it will try to 
        convert its units to the input unit.
        Default : None
    nu_ref : astropy.units.quantity.Quantity
        Reference frequency of the IQU map. Dimensions must match the input 
        map in axis=1, e.g if input map is (3, nside), nu_ref must be (3,).
        Default : None
    label : str
        Map label. Used to name maps.
        Default : None

    """
    if isinstance(input_map, u.quantity.Quantity):
        if unit is not None and unit != input_map.unit:
            input_map.to(unit)
        return IQUMap(input_map=input_map, nu_ref=nu_ref, label=label)


    elif isinstance(input_map, np.ndarray):
        if unit is None:
            input_map *= u.dimensionless_unscaled
        else:
            input_map *= unit
        return IQUMap(input_map=input_map, nu_ref=nu_ref, label=label)

    elif isinstance(input_map, (IQUMap)):
        if unit is not None and unit != input_map.unit:
            input_map.to(unit)
        return input_map

    else:
        raise NotImplemented


class IQUMap:
    """IQU Map object for containing cosmological stokes IQU parameters.

    The object provides a convenient interface for the common IQU map, with
    integrated vectorized arithmatic operations. 
    
    The motivation for such a map object is: 
        1)  to have vectorized operations for IQU maps without having to 
            initialize unpolarized maps with np.zeros in the Q and U 
            parameters (otherwise broadcasting would fail) as this could eat 
            up alot of memory at high nsides.
        2)  to have common map properties, methods and meta data such as P, 
            units, nside, fwhm be predefined attributes and methods of the map
            it self that dynamically update after operations.
    
    TODO: figure out how the map object should behave under arithmetic 
    operations.

    Args:
    -----
    input_map : astropy.units.quantity.Quantity
        I or IQU stokes parameter map.
    nu_ref : astropy.units.quantity.Quantity
        I or IQU reference frequencies.
        Default : None
    label : str
        Map label.
        Default : None

    """
    input_map : u.Quantity
    nu_ref : u.Quantity
    label : str

    @u.quantity_input(nu_ref=(None, u.Hz))
    def __init__(self, input_map, nu_ref=None, label=None):
        self.data = input_map.astype(np.float32)
        if not self._has_pol and self.data.ndim > 1:
            self.data = np.squeeze(self.data)

        self.nu_ref = nu_ref
        self.label = label

        if self.nu_ref is not None:
            if nu_ref.ndim == 1:
                if len(nu_ref) != 3:
                    raise ValueError(
                        'nu_ref must have one value per stokes parameter'
                    )
                    
        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid')


    @property
    def I(self):
        """Returns the stokes I map"""
        if self._has_pol:
            return self.data[0]
        
        return self.data


    @property
    def Q(self):
        """Returns the stokes Q map"""
        if self._has_pol:
            return self.data[1]

        return None


    @property
    def U(self):
        """Returns the I stokes paramater"""
        if self._has_pol:
            return self.data[2]
        
        return None


    @property
    def shape(self):
        """Returns the shape of the IQU map"""
        return self.data.shape


    @property
    def ndim(self):
        """Returns the number of dimensions of the map"""
        return self.data.ndim


    @property
    def unit(self):
        """Returns the units of the IQU map"""
        unit = self.data.unit
        if unit is None:
            return u.dimensionless_unscaled

        return unit


    @property
    def P(self):
        """Polarized map signal. P = sqrt(Q^2 + U^2)"""
        if self._has_pol:
            return np.sqrt(self.Q**2 + self.U**2)

        return None


    @property
    def nside(self):
        """Healpix map resolution"""
        return hp.npix2nside(len(self.I))


    @property
    def _has_pol(self):
        """Returns True if self.Q is not None. False otherwise"""
        if self.data.ndim > 1 and self.data.shape[0] == 3:
            return True

        return False


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
        """If nside is changed, automatically ud_grades I, Q and U maps to the
        new nside.

        Args:
        -----
        new_nside: int
            New Healpix map resolution parameter to ud_grade the maps with.

        """
        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid.')
        if new_nside == self.nside:
            return

        self.data = hp.ud_grade(self.data, new_nside)


    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad))
    def to_fwhm(self, fwhm):
        """Smooths the I, Q and U maps to a given fwhm.
        
        Args:
        -----
        fwhm : astropy.units.quantity.Quantity
            The fwhm of the Gaussian used to smooth the map. Must be either in
            units of arcmin, degrees or radians.

        """
        self.data = hp.smoothing(self.data, fwhm.to(u.rad).value)


    def _arithmetic_operation(self, other, operator):
        """Implements default arithmetic operations for the Map object
        
        Args:
        -----
        other : int, float, np.ndarray, astropy.units.quantity.Quantity
            Object to perform operation with, e.g operator(self, other).
        operator : operator
            Operator function from the default operator library, e.g 
            operator.add, operator.sub, ..

        Returns:
        -------
        Map instance
            A new class instance with values corresponding to the operation.

        """
        if isinstance(other, self.__class__):
            if self._has_pol and other._has_pol:
                input_map = operator(self.data, other.data)
            elif self._has_pol:
                input_map = self.data
                input_map[0] = operator(self.data[0], other.data)
            elif other._has_pol:
                input_map = other.data
                input_map[0] = operator(self.data, other.data[0])
            return self.__class__(
                input_map=input_map,
                nu_ref=self.nu_ref,
            )

        return self.__class__(
            input_map=operator(self.data, other),
            nu_ref=self.nu_ref,
            label=self.label
        )
    

    def __add__(self, other):
        return self._arithmetic_operation(other, operator.add)

    
    def __sub__(self, other):
        return self._arithmetic_operation(other, operator.sub)


    def __mul__(self, other):
        return self._arithmetic_operation(other, operator.mul)


    def __truediv__(self, other):
        return self._arithmetic_operation(other, operator.truediv)


    def __pow__(self, other):
        if isinstance(other, (self.__class__, u.Quantity)):
            input_map = (self.data.value**other.data.value)*self.unit
            return self.__class__(
                input_map=input_map,
                nu_ref=self.nu_ref,
                label=self.label,
            )        
        if isinstance(other, np.ndarray):
            input_map = (self.data.value**other)*self.unit
            return self.__class__(
                input_map=input_map,
                nu_ref=self.nu_ref,
                label=self.label,
            )

        return self._arithmetic_operation(other, operator.pow)



    # Operations commute
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__


    def __iter__(self):
        if self._has_pol:
            return iter(self.data)

        return iter([self.data])


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


    def __repr__(self):
        return f'{self.data}'