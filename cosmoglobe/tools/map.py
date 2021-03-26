import numpy as np
import healpy as hp
import astropy.units as u
import operator


@u.quantity_input(nu_ref=(None, u.Hz))
def to_IQU(input_map, unit=None, nu_ref=None, label=None):
    """Converts a map to a native IQU map object.
    
    Args:
    -----
    input_map : np.ndarray, astropy.units.quantity.Quantity
        Map to be converted.
    unit : astropy.units.Unit
        The units of the input map. If the input map is an astropy object, 
        unit is set the units of the input map. Else, if unit is None, 
        unit is set to u.dimensionless_unscaled.
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
        if input_map.ndim == 1:
            return IQUMap(input_map, 
                          nu_ref=nu_ref,
                          label=label)
        else:
            input_map = input_map
            return IQUMap(I=input_map[0], 
                          Q=input_map[1], 
                          U=input_map[2], 
                          nu_ref=nu_ref,
                          label=label)

    elif isinstance(input_map, np.ndarray):
        if unit is None:
            input_map *= u.dimensionless_unscaled
        else:
            input_map *= unit
        if input_map.ndim == 1:
            return IQUMap(input_map, nu_ref=nu_ref, label=label)
        else:
            return IQUMap(I=input_map[0], 
                          Q=input_map[1], 
                          U=input_map[2],
                          nu_ref=nu_ref,
                          label=label)

    elif isinstance(input_map, (IQUMap)):
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
            parameters (otherwise shapes would not match) as this could eat up
            alot of memory at high nsides.
        2)  to have common map properties and meta data such as P, units, 
            nside, fwhm be predefined attributes that dynamically update after 
            operations.

    TODO: should nu_ref be included as part of the "meta data" of a IQU map?
    
    Args:
    -----
    I : np.ndarray 
        Stokes I map.    
    Q : np.ndarray 
        Stokes U map.
        Default : None   
    U : np.ndarray 
        Stokes Q map. 
        Default : None
    unit : astropy.units.Unit
        Map units.
        Default : None
    nu_ref : astropy.units.quantity.Quantity
        I, Q and U reference frequencies.
        Default : None
    label : str
        Map label.
        Default : None


    """
    I : u.Quantity
    Q : u.Quantity
    U : u.Quantity
    nu_ref : u.Quantity
    label : str

    @u.quantity_input(nu_ref=(None, u.Hz))
    def __init__(self, I, Q=None, U=None, nu_ref=None, label=None):
        """Checks that I, Q, and U are all of the same length. Converts maps 
        to type float32 to save memory.
        """
        self.I = I
        self.Q = Q
        self.U = U
        self.nu_ref = nu_ref
        self.label = label

        self.I = self.I.astype(np.float32)
        if self._has_pol:
            if not (len(self.I) == len(self.Q) == len(self.U)):
                raise ValueError('I, Q, and U must have the same nside')
            
            if not (self.I.unit == self.Q.unit == self.U.unit):
                raise u.UnitsError('I, Q and U must have the same units')

            self.Q = self.Q.astype(np.float32)
            self.U = self.U.astype(np.float32)

        if self.nu_ref is not None and self._has_pol:
            try:
                if len(nu_ref) != 3:
                    raise ValueError('Map is polarized but nu_ref is not')
            except TypeError:
                raise TypeError('Map is polarized but nu_ref is not')

        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid')


    @property
    def data(self):
        """Returns the the map data as a np.ndarray"""
        if self._has_pol:
            return np.array([self.I, self.Q, self.U])

        return self.I


    @property
    def shape(self):
        """Returns the shape of the IQU map"""
        return self.data.shape



    @property
    def unit(self):
        """Returns the units of the IQU map"""
        unit = self.I.unit
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
        if self.Q is not None:
            return True

        return False


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

        self.I = hp.ud_grade(self.I, new_nside)
        self.Q = hp.ud_grade(self.Q, new_nside)
        self.U = hp.ud_grade(self.U, new_nside)


    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad))
    def to_fwhm(self, fwhm):
        """Smooths the I, Q and U maps to a given fwhm.
        
        Args:
        -----
        fwhm : astropy.units.quantity.Quantity
            The fwhm of the Gaussian used to smooth the map. Must be either in
            units of arcmin, degrees or radians.

        """
        self.I = hp.smoothing(self.I, fwhm.to(u.rad).value)
        self.Q = hp.smoothing(self.Q, fwhm.to(u.rad).value)
        self.U = hp.smoothing(self.U, fwhm.to(u.rad).value)


    def _validate_nside(self, other):
        if self.nside != other.nside:
            raise ValueError(
                f'Cant perform operation on maps of different nsides'
            )
    
    def _validate_unit(self, other):
        if self.unit != other.unit:
            raise ValueError(
                f'Cant perform operation on maps of different units'
            )      


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
            self._validate_nside(other)
            if self._has_pol and other._has_pol:
                return self.__class__(
                    I=operator(self.I, other.I),
                    Q=operator(self.Q, other.Q),
                    U=operator(self.U, other.U),
                    unit=self.unit
                )    
            elif self._has_pol:
                return self.__class__(
                    I=operator(self.I, other.I),
                    Q=self.Q,
                    U=self.U,
                    unit=self.unit
                )
            return self.__class__(
                I=operator(self.I, other.I),
                Q=other.Q,
                U=other.U,
                unit=self.unit
            )      

        elif isinstance(other, np.ndarray):
            if self.data.shape != other.shape:
                raise ValueError('Array shapes must match')
            if isinstance(other, u.quantity.Quantity):
                if self.unit != other.unit:
                    raise ValueError(
                        f'Cant perform operation on maps of different units'
                    )   
            if self._has_pol:
                return self.__class__(
                    I=operator(self.I, other[0]),
                    Q=operator(self.Q, other[1]),
                    U=operator(self.U, other[2]),
                    unit=self.unit
                )
            return self.__class__(
                I=operator(self.I, other),
                Q=self.Q,
                U=self.U,
                unit=self.unit
            )
                
        elif isinstance(other, (int, float)):
            if self._has_pol:
                return self.__class__(
                    I=operator(self.I, other),
                    Q=operator(self.Q, other),
                    U=operator(self.U, other),
                    unit=self.unit
                )
            return self.__class__(
                I=operator(self.I, other),
                Q=self.Q,
                U=self.U,
                unit=self.unit
            )
            
        else:
            return NotImplemented         


    def __add__(self, other):
        self._validate_unit
        return self._arithmetic_operation(other, operator.add)

    
    def __sub__(self, other):
        self._validate_unit
        return self._arithmetic_operation(other, operator.sub)


    def __mul__(self, other):
        return self._arithmetic_operation(other, operator.mul)


    def __truediv__(self, other):
        return self._arithmetic_operation(other, operator.truediv)


    def __pow__(self, other):
        if isinstance(other, (self.__class__, u.quantity.Quantity)):
            if other.unit is not None:
                raise u.UnitsError(
                    f'Can only raise something to a dimensionless quantity'
                )
        return self._arithmetic_operation(other, operator.pow)


    # Operations commute
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'{self.data}'