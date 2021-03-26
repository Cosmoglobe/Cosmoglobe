import numpy as np
import healpy as hp
import astropy.units as u
import operator


def to_IQU(input_map):
    """Converts a map to a native IQU map object.
    
    Args:
    -----
    input_map : np.ndarray, astropy.units.quantity,Quantity
        Map to be converted. Must be a np.ndarray or a subclass of np.ndarray.

    """
    if isinstance(input_map, u.quantity.Quantity):
        if input_map.ndim == 1:
            return IQUMap(input_map.value, unit=input_map.unit)
        else:
            unit = input_map.unit
            input_map = input_map.value
            return IQUMap(I=input_map[0], 
                          Q=input_map[1], 
                          U=input_map[2], 
                          unit=unit)

    elif isinstance(input_map, np.ndarray):
        if input_map.ndim == 1:
            return IQUMap(input_map)
        else:
            return IQUMap(I=input_map[0], 
                          Q=input_map[1], 
                          U=input_map[2])

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
    I : np.ndarray
    Q : np.ndarray
    U : np.ndarray
    unit : u.Unit
    nu_ref : u.Quantity
    label : str

    @u.quantity_input(nu_ref=(None, u.Hz))
    def __init__(self, I, Q=None, U=None, unit=None, nu_ref=None, label=None):
        """Checks that I, Q, and U are all of the same length. Converts maps 
        to type np.float.32 to save memory.
        """
        self.I, self.Q, self.U = I, Q, U
        if self._has_pol:
            if not (len(self.I) == len(self.Q) == len(self.U)):
                raise ValueError('I, Q, and U must have the same nside.')
            self.I = self.I.astype(np.float32)
            self.Q = self.Q.astype(np.float32)
            self.U = self.U.astype(np.float32)
        else:
            self.I = self.I.astype(np.float32)

        if unit is None:
            self.unit = u.dimensionless_unscaled
        else:
            self.unit = unit

        if nu_ref is not None:
            if self._has_pol:
                if len(nu_ref) != 3:
                    raise ValueError('Map is polarized but nu_ref is not')
                
        self.nu_ref = nu_ref



        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid.')


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

    @property
    def mask(self, mask):
        """Applies a mask to the data"""

        sigs = ["I"]
        if self._has_pol:
            sigs += ["Q", "U"]

        for sig in self:
            sig = hp.ma(sig)
            print(sig)
            setattr(self, sig, hp.ma(getattr(self,sig))    
            
            self.[sig].mask = np.logical_not(mask)

            m.mask = np.logical_not(mask)

        # Don't know what this does, from paperplots by Zonca.
        grid_mask = m.mask[grid_pix]
        grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)

"""
def apply_mask(m, mask, grid_pix, mfill, polt, cmap):
    click.echo(click.style(f"Masking using {mask}", fg="yellow"))
    # Apply mask
    hp.ma(m)
    mask_field = polt-3 if polt>2 else polt
    m.mask = np.logical_not(hp.read_map(mask, field=mask_field, verbose=False, dtype=None))

    # Don't know what this does, from paperplots by Zonca.
    grid_mask = m.mask[grid_pix]
    grid_map = np.ma.MaskedArray(m[grid_pix], grid_mask)

    if mfill:
        cmap.set_bad(mfill)  # color of missing pixels
        # cmap.set_under("white") # color of background, necessary if you want to use
        # using directly matplotlib instead of mollview has higher quality output

    return grid_map, cmap
    
def remove_md(m, remove_dipole, remove_monopole, nside):        
    if remove_monopole:
        dip_mask_name = remove_monopole
    if remove_dipole:
        dip_mask_name = remove_dipole
    # Mask map for dipole estimation
    if dip_mask_name == 'auto':
        mono, dip = hp.fit_dipole(m, gal_cut=30)
    else:
        m_masked = hp.ma(m)
        m_masked.mask = np.logical_not(hp.read_map(dip_mask_name,verbose=False,dtype=None,))

        # Fit dipole to masked map
        mono, dip = hp.fit_dipole(m_masked)

    # Subtract dipole map from data
    if remove_dipole:
        click.echo(click.style("Removing dipole:", fg="yellow"))
        click.echo(click.style("Dipole vector:",fg="green") + f" {dip}")
        click.echo(click.style("Dipole amplitude:",fg="green") + f" {np.sqrt(np.sum(dip ** 2))}")

        # Create dipole template
        nside = int(nside)
        ray = range(hp.nside2npix(nside))
        vecs = hp.pix2vec(nside, ray)
        dipole = np.dot(dip, vecs)
        
        m = m - dipole
    if remove_monopole:
        click.echo(click.style("Removing monopole:", fg="yellow"))
        click.echo(click.style("Mono:",fg="green") + f" {mono}")
        m = m - mono
    return m
"""


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

    def __iter__(self):
         if self._has_pol:
            return iter([self.I, self.Q, self.U,])

        return iter([self.I,])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'{self.data}'