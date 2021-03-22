from dataclasses import dataclass, field
import numpy as np
import healpy as hp
import astropy.units as u


class Component:
    """
    Base class for all sky components.

    Any model you make should subclass this class.

    """

    def __init__(self, amp, params, *args, **kwargs):
        self.amp = amp
        self.params = params


    def get_freq_scaling(self, nu):
        return 1



    

@dataclass(eq=False)
class Map:
    """IQU Map object for containing cosmological stokes IQU parameters.

    The object provides a convenient way to refer to a a maps stokes parameters
    in addition to vectorized arithmatic operations on all three stokes parameters.
    
    Args:
    -----
    I : np.ndarray 
        Stokes I map.    
    Q : np.ndarray 
        Stokes U map.    
    U : np.ndarray 
        Stokes Q map. 
    unit: astropy.units.Unit
        Map units.

    """
    I : np.ndarray
    Q : np.ndarray = None
    U : np.ndarray = None
    unit : u.Unit = None


    def __post_init__(self):
        """
        Cheks that I, Q, and U are all
        """
        if not (len(self.I) == len(self.Q) == len(self.U)):
            raise ValueError('I, Q, and U must have the same shape.')

        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid.')

        self.I = self.I.astype(np.float32)
        self.Q = self.I.astype(np.float32)
        self.U = self.I.astype(np.float32)


    @property
    def P(self):
        """
        Polarized map signal, P = sqrt(Q^2 + U^2)

        """
        return np.sqrt(self.Q**2 + self.U**2)


    @property
    def nside(self):
        """
        Healpix map resolution.

        """
        return hp.npix2nside(len(self.I))


    def _validate_nside_unit(self, other, operation):
        if self.nside != other.nside:
            raise ValueError(f'Cant {operation} maps of different nsides')
        if self.unit != other.unit:
            raise ValueError(f'Cant {operation} maps of different units')      


    def __add__(self, other):
        if isinstance(other, Map):
            self._validate_nside_unit(other, 'add')
            return Map(I=(self.I + other.I),
                       Q=(self.Q + other.Q),
                       U=(self.U + other.U),
                       unit=self.unit)    
        
        elif isinstance(other, (int, float)):
            return Map(I=(self.I + other),
                       Q=(self.Q + other),
                       U=(self.U + other),
                       unit=self.unit)

        elif isinstance(other, np.ndarray):
            if len(self.I) != len(other):
                raise ValueError('Lenght of arrays are not matching')
            return Map(I=(self.I + other),
                       Q=(self.Q + other),
                       U=(self.U + other),
                       unit=self.unit)

        else:
            return NotImplemented       


    def __sub__(self, other):
        if isinstance(other, Map):
            self._validate_nside_unit(other, 'subtract')
            return Map(I=(self.I - other.I),
                       Q=(self.Q - other.Q),
                       U=(self.U - other.U),
                       unit=self.unit)    
        
        elif isinstance(other, (int, float)):
            return Map(I=(self.I - other),
                       Q=(self.Q - other),
                       U=(self.U - other),
                       unit=self.unit)

        elif isinstance(other, np.ndarray):
            if len(self.I) != len(other):
                raise ValueError('Lenght of arrays are not matching')
            return Map(I=(self.I - other),
                       Q=(self.Q - other),
                       U=(self.U - other),
                       unit=self.unit) 

        else:
            return NotImplemented  


    def __mul__(self, other):
        if isinstance(other, Map):
            self._validate_nside_unit(other, 'multiply')
            return Map(I=(self.I * other.I),
                       Q=(self.Q * other.Q),
                       U=(self.U * other.U),
                       unit=self.unit)    
        
        elif isinstance(other, (int, float)):
            return Map(I=(self.I * other),
                       Q=(self.Q * other),
                       U=(self.U * other),
                       unit=self.unit)

        elif isinstance(other, np.ndarray):
            if len(self.I) != len(other):
                raise ValueError('Lenght of arrays are not matching')
            return Map(I=(self.I * other),
                       Q=(self.Q * other),
                       U=(self.U * other),
                       unit=self.unit)

        else:
            return NotImplemented  


    def __truediv__(self, other):
        if isinstance(other, Map):
            self._validate_nside_unit(other, 'divide')
            return Map(I=(self.I / other.I),
                       Q=(self.Q / other.Q),
                       U=(self.U / other.U),
                       unit=self.unit)    
        
        elif isinstance(other, (int, float)):
            return Map(I=(self.I / other),
                       Q=(self.Q / other),
                       U=(self.U / other),
                       unit=self.unit)

        elif isinstance(other, np.ndarray):
            if len(self.I) != len(other):
                raise ValueError('Lenght of arrays are not matching')
            return Map(I=(self.I / other),
                       Q=(self.Q / other),
                       U=(self.U / other),
                       unit=self.unit)

        else:
            return NotImplemented  


@dataclass
class Parameters:
    nside : int
    nu_ref : u.GHz 
    unit : u.Unit = u.uK
    spectral : tuple = field(default_factory=tuple)


    def __post_init__(self):
        if not hp.isnsideok(self.nside, nest=True):
            raise ValueError(f'nside: {self.nside} is not valid.')


if __name__ == '__main__':
    comp = Component()
    # print(comp.parameters)
    a = Map(I=np.arange(hp.nside2npix(64)), 
            Q=np.arange(hp.nside2npix(64)),
            U=np.arange(hp.nside2npix(64)),
            unit=u.uK)    
    b = Map(I=np.arange(hp.nside2npix(64)), 
            Q=np.arange(hp.nside2npix(64)),
            U=np.arange(hp.nside2npix(64)),
            unit=u.uK)

    comp = Component()
    print(comp.parameters)
