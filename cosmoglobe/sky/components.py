from ..tools.bandpass import (
    _get_normalized_bandpass, 
    _get_interp_parameters, 
    _get_unit_conversion,
    _interp1d,
    _interp2d,
)
from ..science.functions import (
    blackbody_emission, 
    gaunt_factor, 
    cmb_to_brightness
)
from .. import data as data_dir

import astropy.units as u
import numpy as np
import healpy as hp
import pathlib


class Component:
    """Base class for all sky components.

    Any sky component you make should subclass this class. All components must
    implement the get_freq_scaling method. This method needs to return the
    frequency scaling factor from the reference frequency of the amplitude
    template to a arbritrary frequency. Following is an example of a custom 
    implementation of a component whos emission scales as a simple power law::

        import cosmoglobe.sky as sky

        class PowerLaw(sky.Component):
            def __init__(self, comp_name, amp, freq_ref, beta):
                super().__init__(comp_name, amp, freq_ref, beta=beta)

            def get_freq_scaling(self, freq, freq_ref, beta):
                return (freq/freq_ref)**beta

    A component is essentially defined by its get_freq_scaling function, 
    which it is required to implement. This function is required to take in 
    a freq and a freq_ref at minimum (even if it is not required to evalute 
    the emission). Further, the get_freq_scaling function can take in any 
    number of spectral parameters, although a model with more than two spectral
    parameters that varies across the sky is not supported under bandpass
    integration.

    Args:
    -----
    comp_name (str):
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
    def __init__(self, comp_name, amp, freq_ref, **spectral_parameters):
        self.comp_name = comp_name
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
    def get_emission(self, freq, bandpass=None, output_unit=None):
        """Returns the component emission at an arbitrary frequency or
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
        # Expand dimension on 1d spectral parameters from from (n,) to (n, 1) 
        # to support broadcasting
        spectral_parameters = {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in self.spectral_parameters.items()
        }

        if bandpass is not None:
            unit_conversion_factor = (
                _get_unit_conversion(bandpass, freq, output_unit, input_unit)
            )
            bandpass = _get_normalized_bandpass(bandpass, freq, input_unit)
            bandpass_conversion_factor = (
                self._get_bandpass_conversion(freq, freq_ref, bandpass, 
                                              spectral_parameters)
            )
            return self.amp*bandpass_conversion_factor*unit_conversion_factor

        else:
            if freq.ndim > 0:
                scaling = (self.get_freq_scaling(freq, freq_ref, 
                                                 **spectral_parameters)
                           for freq in freq)
            else:
                scaling = self.get_freq_scaling(freq, freq_ref, 
                                                **spectral_parameters)

        if input_unit != output_unit:
            return (self.amp*scaling).to(
                output_unit, equivalencies=u.brightness_temperature(freq)
            )
        return self.amp*scaling


    def _get_bandpass_conversion(self, freqs, freq_ref, bandpass, 
                                 spectral_parameters, n=20):
        """Returns the frequency scaling factor given a frequency array and a 
        bandpass profile.

        For more information on the computations, see section 4.2 in 
        https://arxiv.org/abs/2011.05609.

        Args:
        -----
        freqs (`astropy.units.Quantity`):
            Frequencies corresponding to the bandpass weights.
        freq_ref (`numpy.ndarray`):
            Reference frequencies for the amplitude map.
        bandpass (`astropy.units.Quantity`):
            Normalized bandpass profile. Must have signal units.
        spectral_parameters (dict):
            Spectral parameters required to compute the frequency scaling factor. 
        n (int):
            Number of points in the regular interpolation grid. Default: 20

        Returns:
        --------
        float, `numpy.ndarray`
            Frequency scaling factor given a bandpass.

        """
        interp_parameters = _get_interp_parameters(spectral_parameters)
        if not interp_parameters:
            freq_scaling = self.get_freq_scaling(freqs, freq_ref, 
                                                 **spectral_parameters)
            # Reshape to support broadcasting for comps where freq_ref = None 
            # e.g cmb
            if freq_scaling.ndim > 1:
                return np.expand_dims(
                    np.trapz(freq_scaling*bandpass, freqs), axis=1
                )
            return np.trapz(freq_scaling*bandpass, freqs)

        elif len(interp_parameters) == 1:
            return _interp1d(self, bandpass, freqs, freq_ref, 
                             interp_parameters, spectral_parameters.copy())

        elif len(interp_parameters) == 2:
            return _interp2d(self, bandpass, freqs, freq_ref, 
                             interp_parameters, spectral_parameters.copy())

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
        if new_nside == self.amp.nside:
            return
        if not hp.isnsideok(new_nside, nest=True):
            raise ValueError(f'nside: {new_nside} is not valid.')

        self.amp.to_nside(new_nside)
        for key, val in self.spectral_parameters.items():
            try:
                self.spectral_parameters[key] = hp.ud_grade(val.value, 
                                                            new_nside)*val.unit
            except AttributeError:
                self.spectral_parameters[key] = hp.ud_grade(val, new_nside)


    @property
    def _is_polarized(self):
        """Returns True if component is polarized and False if not"""
        if self.amp.shape[0] == 3:
            return True
        return False


    def __repr__(self):
        main_repr = f'{self.__class__.__name__}'
        main_repr += '(amp, freq_ref, '
        for spectral in self.spectral_parameters:
            main_repr += f'{spectral}, '
        main_repr = main_repr[:-2]
        main_repr += ')'

        return main_repr



class PowerLaw(Component):
    """PowerLaw component class. Represents any component with a frequency 
    scaling given by a simple power law.

    Args:
    -----
    comp_name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude map in units of Hertz.
    beta (`numpy.ndarray`, `astropy.units.Quantity`):
        The power law spectral index. The spectral index can vary over the sky, 
        and is therefore commonly given as a shape (3, nside) array, but it can 
        take the value of a scalar.

    """
    def __init__(self, comp_name, amp, freq_ref, beta):
        super().__init__(comp_name, amp, freq_ref, beta=beta)


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
    """Modified blackbody component class. Represents any component with a 
    frequency scaling given by a simple power law times a blackbody.

    Args:
    -----
    comp_name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude map in units of Hertz.
    beta (`numpy.ndarray`, `astropy.units.Quantity`):
        The power law spectral index. The spectral index can vary over the sky, 
        and is therefore commonly given as a shape (3, nside) array, but it can 
        take the value of a scalar.
    T (`astropy.units.Quantity`):
        Temperature map of the blackbody with unit K and shape (nside,). Can 
        also take the value of a scalar similar to beta.

    """
    def __init__(self, comp_name, amp, freq_ref, beta, T):
        super().__init__(comp_name, amp, freq_ref, beta=beta, T=T)


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
        blackbody_ratio = blackbody_emission(freq, T) / blackbody_emission(freq_ref, T)
        scaling = (freq/freq_ref)**(beta-2) * blackbody_ratio
        return scaling



class LinearOpticallyThinBlackBody(Component):
    """Linearized optically thin blackbody emission component class. Represents 
    a component with a frequency scaling given by a linearized optically thin 
    blacbody spectrum, strictly only valid in the optically thin case (tau << 1).

    TODO: find a more general name for this class

    Args:
    -----
    comp_name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude map in units of Hertz.
    Te (`astropy.units.Quantity`):
        Electron temperature map with unit K.

    """
    def __init__(self, comp_name, amp, freq_ref, Te):
        super().__init__(comp_name, amp, freq_ref, Te=Te)


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
    """Spinning dust component class using a template from the SpDust2 code.
    For more info, please see the following papers: 
        - Ali-Haïmoud et al. (2009)
        - Ali-Haimoud (2010)
        - Silsbee et al. (2011)

    TODO: find a better solution to reading in data without importing the 
          entire data module.

    Args:
    -----
    comp_name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude map in units of Hertz.
    nu_p (`astropy.units.Quantity`):
        Peak frequency.

    """
    def __init__(self, comp_name, amp, freq_ref, nu_p):
        super().__init__(comp_name, amp, freq_ref, nu_p=nu_p)
        spdust2_freq, spdust2_amp = np.loadtxt(
            pathlib.Path(data_dir.__path__[0]) / 'spdust2_cnm.dat', unpack=True
        )
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
        peak_scale = (30*u.GHz / nu_p)
        interp = np.interp((freq*peak_scale).si.value, spdust2[0], spdust2[1])
        interp_ref = np.interp((freq_ref*peak_scale).si.value, spdust2[0], spdust2[1])

        scaling = interp/interp_ref
        return scaling



class BlackBodyCMB(Component):
    """Blackbody CMB component class. Represents blackbody emission of the CMB
    converted from units of K_CMB to K_RJ.

    TODO: find a more suiting name for this component.

    Args:
    -----
    comp_name (str):
        Name/label of the component. Is used to set the component attribute 
        in a `cosmoglobe.sky.Model`.
    amp (`numpy.ndarray`, `astropy.units.Quantity`, `cosmoglobe.StokesMap`):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (`astropy.units.Quantity`):
        Reference frequencies for the amplitude map in units of Hertz.

    """
    def __init__(self, comp_name, amp):
        super().__init__(comp_name, amp, freq_ref=None)


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
        return cmb_to_brightness(freq)