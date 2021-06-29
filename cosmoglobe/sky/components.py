from cosmoglobe.utils.bandpass import (
    get_bandpass_coefficient,
    get_interp_parameters, 
    get_normalized_bandpass, 
    interp1d,
    interp2d,
)
from cosmoglobe.utils.functions import (
    blackbody_emission,
    brightness_to_thermodynamical, 
    gaunt_factor, 
    thermodynamical_to_brightness
)
from cosmoglobe.utils.utils import _get_astropy_unit

from pathlib import Path
from sys import exit
import astropy.units as u
import numpy as np
import healpy as hp

class Component:
    """Base class for a sky component from the Cosmoglobe Sky Model.

    A component is defined by its get_freq_scaling function, which it is 
    required to implement. This function is required to take in a freq and a 
    freq_ref at minimum (even if it is not required to evalute the emission). 
    Further, the get_freq_scaling function can take in any number of spectral 
    parameters, although a model with more than two spectral parameters that 
    varies across the sky is currently not supported under bandpass integration.

    Args:
    -----
    name (str):
        The name or label of the component. The label becomes the attribute 
        name of the component in a `cosmoglobe.sky.Model`.
    amp (`astropy.units.Quantity`):
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude template in units of Hertz.
        The input must be an astropy quantity containing the the reference 
        frequency for the stokes I amplitude template, and optionally the
        reference frequency for the soktes Q and U templates if the component 
        is polarized. Example: freq_ref=freq_ref_I*u.GHz, or 
        freq_ref=[freq_ref_I, freq_ref_P]*u.GHz
    spectral_parameters (dict):
        Spectral parameters required to compute the frequency scaling factor. 
        These can be scalars, numpy ndarrays or astropy quantities. 
        Default: None

    """
    def __init__(self, name, amp, freq_ref, **spectral_parameters):
        self.name = name
        self.amp = amp
        self.freq_ref = self._set_freq_ref(freq_ref)
        self.spectral_parameters = spectral_parameters


    @staticmethod
    @u.quantity_input(freq=u.Hz)
    def _set_freq_ref(freq_ref):
        """Reshapes the freq_ref into a broadcastable shape"""
        if freq_ref is None:
            return
        elif freq_ref.ndim == 0:
            return freq_ref
        elif freq_ref.ndim == 1:
            return np.expand_dims(
                u.Quantity([freq_ref[0], freq_ref[1], freq_ref[1]]), axis=1
            )
        else:
            raise ValueError('freq_ref must have a maximum len of 2')


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def __call__(self, freq, bandpass=None, output_unit=None):
        """Simulates the component emission at an arbitrary frequency or
        integrated over a bandpass.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            A frequency, or a list of frequencies at which to evaluate the 
            component emission. If a corresponding bandpass is not supplied, 
            a delta peak in frequency is assumed.
        bandpass (`astropy.units.Quantity`):
            Bandpass profile in signal units. The shape of the bandpass must
            match that of the freq input. Default : None
        output_unit (`astropy.units.Unit`):
            The desired output unit of the emission. Must be signal units. 
            Default: None

        Returns
        -------
        (`astropy.units.Quantity`):
            Component emission.

        """
        freq_ref = self.freq_ref
        input_unit = self.amp.unit

        # Expand dimension on rank-1 arrays from from (n,) to (n, 1) to support
        # broadcasting with (1, nside) or (3, nside) arrays
        amp = self.amp if self.amp.ndim != 1 else np.expand_dims(self.amp, axis=0)
        spectral_parameters = {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in self.spectral_parameters.items()
        }

        #Assuming delta frequencies
        if bandpass is None:
            if freq.ndim == 0:
                scaling = self.get_freq_scaling(
                    freq, freq_ref, **spectral_parameters
                )
            else:
                scaling = (
                    self.get_freq_scaling(freq, freq_ref, **spectral_parameters)
                    for freq in freq
                )
            emission = amp*scaling
            
            if output_unit is not None:
                try:
                    output_unit = u.Unit(output_unit)
                except ValueError:
                    if output_unit.lower().endswith('k_rj'):
                        output_unit = u.Unit(output_unit[:-3])
                        return emission.to(output_unit)
                    elif output_unit.lower().endswith('k_cmb'):
                        output_unit = u.Unit(output_unit[:-4])   
                        emission *= brightness_to_thermodynamical(freq)
                        return emission.to(output_unit)

                emission = emission.to(
                    output_unit, equivalencies=u.brightness_temperature(freq)
                )

        # Perform bandpass integration
        else:
            bandpass = get_normalized_bandpass(bandpass, freq, input_unit)
            bandpass_coefficient = get_bandpass_coefficient(
                bandpass, freq, output_unit
            )
            bandpass_scaling = self._get_bandpass_scaling(freq, bandpass)

            emission = amp*bandpass_scaling*bandpass_coefficient

        return emission.to(_get_astropy_unit(output_unit))


    def _get_bandpass_scaling(self, freqs, bandpass):
        """Returns the frequency scaling factor given a bandpass profile and a
        corresponding frequency array.

        Args:
        -----
        freqs (`astropy.units.Quantity`):
            Bandpass profile frequencies.
        bandpass (`astropy.units.Quantity`):
            Normalized bandpass profile.

        Returns:
        --------
        float, `numpy.ndarray`
            Frequency scaling factor given a bandpass.

        """
        interp_parameters = get_interp_parameters(self.spectral_parameters)

        # Component does not have any spatially varying spectral parameters
        if not interp_parameters:
            freq_scaling = self.get_freq_scaling(
                freqs, self.freq_ref, **self.spectral_parameters
            )
            # Reshape to support broadcasting for comps where freq_ref = None 
            # e.g cmb
            if freq_scaling.ndim > 1:
                return np.expand_dims(
                    np.trapz(freq_scaling*bandpass, freqs), axis=1
                )
            return np.trapz(freq_scaling*bandpass, freqs)

        # Component has one sptatially varying spectral parameter
        elif len(interp_parameters) == 1:
            return interp1d(
                self, freqs, bandpass, interp_parameters, 
                self.spectral_parameters.copy()
            )

        # Component has two sptatially varying spectral parameter
        elif len(interp_parameters) == 2:
            return interp2d(
                self, freqs, bandpass, interp_parameters, 
                self.spectral_parameters.copy()
            )

        else:
            raise NotImplementedError(
                'Bandpass integration for comps with more than two spectral '
                'parameters is not implemented'
            )


    def to_nside(self, new_nside):
        """ud_grades all maps in the component to a new nside.

        Args:
        -----
        new_nside (int):
            Healpix map resolution parameter.

        """
        nside = hp.get_nside(self.amp)
        if new_nside == nside:
            print(f'Model is already at nside {nside}')
            return
        if not hp.isnsideok(new_nside, nest=True):
            raise ValueError(f'nside: {new_nside} is not valid.')

        self.amp = hp.ud_grade(self.amp.value, new_nside)*self.amp.unit
        for key, val in self.spectral_parameters.items():
            if hp.nside2npix(nside) in np.shape(val):
                try:
                    self.spectral_parameters[key] = hp.ud_grade(val.value, 
                                                            new_nside)*val.unit
                except AttributeError:
                    self.spectral_parameters[key] = hp.ud_grade(val, new_nside)


    @property
    def is_polarized(self):
        """Returns True if component is polarized and False if not"""
        if self.amp.shape[0] == 3:
            return True
        return False


    def __repr__(self):
        main_repr = f'{self.__class__.__name__}'
        main_repr += '('
        extra_repr = ''
        for key in self.spectral_parameters.keys():
            extra_repr += f'{key}, '
        if extra_repr:
            extra_repr = extra_repr[:-2]
        main_repr += extra_repr
        main_repr += ')'

        return main_repr



class PowerLaw(Component):
    """PowerLaw component class. Represents Synchrotron emission in the 
    Cosmoglobe Sky Model.

    Args:
    -----
    name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.Model`.
    amp (`astropy.units.Quantity`):
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude template in units of Hertz.
        The input must be an astropy quantity containing the the reference 
        frequency for the stokes I amplitude template, and optionally the
        reference frequency for the stokes Q and U templates if the component 
        is polarized. Example: freq_ref=freq_ref_I*u.GHz, or 
        freq_ref=[freq_ref_I, freq_ref_P]*u.GHz
    beta (`numpy.ndarray`, `astropy.units.Quantity`):
        The power law spectral index. The spectral index can vary over the sky, 
        and is therefore commonly given as a shape (3, nside) array, but it can 
        take the value of a scalar.

    """
    def __init__(self, name, amp, freq_ref, beta):
        super().__init__(name, amp, freq_ref, beta=beta)


    def get_freq_scaling(self, freq, freq_ref, beta):
        """Computes the frequency scaling from the reference frequency freq_ref 
        to an arbitrary frequency, which depends on the spectral parameter
        beta.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            Frequency at which to evaluate the model.
        freq_ref (`astropy.units.Quantity`):
            Reference frequencies for the amplitude map.
        beta (`numpy.ndarray`, `astropy.units.Quantity`):
            The power law spectral index.
            
        Returns:
        --------
        scaling (`astropy.units.Quantity`):
            Frequency scaling factor with dimensionless units.

        """
        scaling = (freq/freq_ref)**beta
        return scaling


class ModifiedBlackBody(Component):
    """Modified blackbody component class. Represents thermal dust in the
    Cosmoglobe Sky Model.

    Args:
    -----
    name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`astropy.units.Quantity`):
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude template in units of Hertz.
        The input must be an astropy quantity containing the the reference 
        frequency for the stokes I amplitude template, and optionally the
        reference frequency for the soktes Q and U templates if the component 
        is polarized. Example: freq_ref=freq_ref_I*u.GHz, or 
        freq_ref=[freq_ref_I, freq_ref_P]*u.GHz
    beta (`numpy.ndarray`, `astropy.units.Quantity`):
        The power law spectral index. The spectral index can vary over the sky, 
        and is therefore commonly given as a shape (3, nside) array, but it can 
        take the value of a scalar.
    T (`astropy.units.Quantity`):
        Temperature map of the blackbody with unit K and shape (nside,). Can 
        also take the value of a scalar similar to beta.

    """
    def __init__(self, name, amp, freq_ref, beta, T):
        super().__init__(name, amp, freq_ref, beta=beta, T=T)


    def get_freq_scaling(self, freq, freq_ref, beta, T):
        """Computes the frequency scaling from the reference frequency freq_ref 
        to an arbitrary frequency, which depends on the spectral parameters
        beta and T.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            Frequency at which to evaluate the model.
        freq_ref (`astropy.units.Quantity`):
            Reference frequencies for the amplitude map.
        beta (`numpy.ndarray`, `astropy.units.Quantity`):
            The power law spectral index.
        T (`astropy.units.Quantity`): 
            Temperature of the blackbody.
            
        Returns:
        --------
        scaling (`astropy.units.Quantity`):
            Frequency scaling factor with dimensionless units.

        """
        blackbody_ratio = (
            blackbody_emission(freq, T) / blackbody_emission(freq_ref, T)
        )
        scaling = (freq/freq_ref)**(beta-2) * blackbody_ratio
        return scaling



class FreeFree(Component):
    """FreeFree emission component class. Represents FreeFree in the
    Cosmoglobe Sky Model

    Args:
    -----
    name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`astropy.units.Quantity`):
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude template in units of Hertz.
        The input must be an astropy quantity containing the the reference 
        frequency for the stokes I amplitude template, and optionally the
        reference frequency for the soktes Q and U templates if the component 
        is polarized. Example: freq_ref=freq_ref_I*u.GHz, or 
        freq_ref=[freq_ref_I, freq_ref_P]*u.GHz
    Te (`astropy.units.Quantity`):
        Electron temperature map with unit K.

    """
    def __init__(self, name, amp, freq_ref, Te):
        super().__init__(name, amp, freq_ref, Te=Te)


    def get_freq_scaling(self, freq, freq_ref, Te):
        """Computes the frequency scaling from the reference frequency freq_ref 
        to an arbitrary frequency, which depends on the spectral parameter Te.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            Frequency at which to evaluate the model.
        freq_ref (`astropy.units.Quantity`):
            Reference frequencies for the amplitude map.
        Te (`astropy.units.Quantity`): 
            Electron temperature.
            
        Returns:
        --------
        scaling (`astropy.units.Quantity`):
            Frequency scaling factor with dimensionless units.

        """
        gaunt_factor_ratio = gaunt_factor(freq, Te) / gaunt_factor(freq_ref, Te)
        scaling = (freq_ref/freq)**2 * gaunt_factor_ratio
        return scaling



class SpDust2(Component):
    """Spinning dust component class using a precomputed template from the
    SpDust2 code to interpolate.
    For more info, please see the following papers: 
        - Ali-Haïmoud et al. (2009)
        - Ali-Haimoud (2010)
        - Silsbee et al. (2011)

    TODO: find a better solution to reading in data without importing the 
          entire data module.

    Args:
    -----
    name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`astropy.units.Quantity`):
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude template in units of Hertz.
        The input must be an astropy quantity containing the the reference 
        frequency for the stokes I amplitude template, and optionally the
        reference frequency for the soktes Q and U templates if the component 
        is polarized. Example: freq_ref=freq_ref_I*u.GHz, or 
        freq_ref=[freq_ref_I, freq_ref_P]*u.GHz
    nu_p (`astropy.units.Quantity`):
        Peak frequency.

    """
    def __init__(self, name, amp, freq_ref, nu_p):
        super().__init__(name, amp, freq_ref, nu_p=nu_p)

        # Read in spdust2 template
        DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
        SPDUST2_FILE = DATA_DIR / 'spdust2_cnm.dat'
        spdust2_freq, spdust2_amp = np.loadtxt(SPDUST2_FILE, unpack=True)
        spdust2_freq = u.Quantity(spdust2_freq, unit=u.GHz)
        spdust2_amp = u.Quantity(spdust2_amp, unit=(u.Jy/u.sr)).to(
            u.K, equivalencies=u.brightness_temperature(spdust2_freq)
        )        
        self.spdust2 = np.array([spdust2_freq.si.value, spdust2_amp.si.value])


    def get_freq_scaling(self, freq, freq_ref, nu_p):
        """Computes the frequency scaling from the reference frequency freq_ref 
        to an arbitrary frequency, which depends on the spectral parameter nu_p.

        Args:
        -----
        freq (`astropy.units.Quantity`):
            Frequency at which to evaluate the model.
        freq_ref (`astropy.units.Quantity`):
            Reference frequencies for the amplitude map.
        nu_p (`astropy.units.Quantity`): 
            Electron temperature.
            
        Returns:
        --------
        scaling (`astropy.units.Quantity`):
            Frequency scaling factor with dimensionless units.

        """
        spdust2 = self.spdust2

        peak_scale = 30*u.GHz / nu_p
        interp = np.interp((freq*peak_scale).si.value, spdust2[0], spdust2[1])
        interp_ref = (
            np.interp((freq_ref*peak_scale).si.value, spdust2[0], spdust2[1])
        )
        scaling = interp/interp_ref
        return scaling



class CMB(Component):
    """CMB component class. Assumes that the component amplitude template is in
    units of K_CMB. The emission is defined as the conversion between K_CMB and
    K_RJ units.

    Args:
    -----
    name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    """
    def __init__(self, name, amp):
        super().__init__(name, amp, freq_ref=None)


    def get_freq_scaling(self, freq, freq_ref):
        """Computes the frequency scaling from K_CMB to K_RJ as a frequency.
        Args:
        -----
        freq (`astropy.units.Quantity`):
            Frequency at which to evaluate the model.
            
        Returns:
        --------
        scaling (`astropy.units.Quantity`):
            Frequency scaling factor with dimensionless units.
        """
        return thermodynamical_to_brightness(freq)