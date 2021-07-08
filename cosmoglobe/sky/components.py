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
from cosmoglobe.utils.utils import _get_astropy_unit, gaussian_beam_2D

from pathlib import Path
from sys import exit
import warnings
from tqdm import tqdm
import astropy.units as u
import numpy as np
import healpy as hp
import sys

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


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
    def __init__(self, amp, freq_ref, **spectral_parameters):
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


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None), 
                      fwhm=(u.rad, u.deg, u.arcmin))
    def __call__(self, freq, bandpass=None, fwhm=0.0*u.rad, output_unit=u.uK):
        r"""Computes the simulated component emission at an arbitrary frequency 
        or integrated over a bandpass.

        .. math::

            \mathbf{s}_\mathrm{comp} = \mathbf{a}_\mathrm{comp} \; 
            \mathrm{scaling}_\mathrm{comp}

        where :math:`\mathbf{a}_\mathrm{comp}` is the amplitude template of 
        the component at some reference frequency, and 
        :math:`\mathrm{scaling}_\mathrm{comp}` is the scaling factor for 
        component.


        Parameters
        ----------
        freq : `astropy.units.Quantity`
            A frequency, or a list of frequencies at which to evaluate the 
            component emission. If a corresponding bandpass is not supplied, 
            a delta peak in frequency is assumed.
        bandpass : `astropy.units.Quantity`
            Bandpass profile in signal units. The shape of the bandpass must
            match that of the freq input. Default : None
        output_unit : `astropy.units.Unit`
            The desired output unit of the emission. Must be signal units. 
            Default: None
        fwhm : `astropy.units.Quantity`
            The full width half max parameter of the beam. Default: None

        Returns
        -------
        `astropy.units.Quantity`
            Component emission.

        """
        freq_ref = self.freq_ref
        input_unit = u.Unit('uK')

        # Expand dimension on rank-1 arrays from from (n,) to (n, 1) to support
        # broadcasting with (1, nside) or (3, nside) arrays
        amp = self.amp if self.amp.ndim != 1 else np.expand_dims(self.amp, axis=0)
        spectral_parameters = {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in self.spectral_parameters.items()
        }

        #Assuming delta frequencies
        if bandpass is None:
            if freq.size == 1:
                scaling = self.get_freq_scaling(
                    freq, freq_ref, **spectral_parameters
                )
            else:
                scaling = (
                    self.get_freq_scaling(freq, freq_ref, **spectral_parameters)
                    for freq in freq
                )
            if self.diffuse:
                emission = amp*scaling
                if fwhm.value != 0.0:
                    if self.is_polarized:
                        emission = hp.smoothing(
                            emission, fwhm=fwhm.to(u.rad).value
                        )*emission.unit
                    else:
                        emission[0] = hp.smoothing(
                            emission[0], fwhm=fwhm.to(u.rad).value
                        )*emission.unit    

            else:
                # self.amp is not a healpix map for non diffuse comps
                emission = self.get_map(amp=amp*scaling, fwhm=fwhm)
            
            if output_unit is not None:
                try:
                    output_unit = u.Unit(output_unit)
                    emission = emission.to(
                        output_unit, equivalencies=u.brightness_temperature(freq)
                    )
                except ValueError:
                    if output_unit.lower().endswith('k_rj'):
                        output_unit = u.Unit(output_unit[:-3])
                    elif output_unit.lower().endswith('k_cmb'):
                        output_unit = u.Unit(output_unit[:-4])   
                        emission *= brightness_to_thermodynamical(freq)
                
            return emission

        # Perform bandpass integration
        else:
            bandpass = get_normalized_bandpass(bandpass, freq, input_unit)
            bandpass_coefficient = get_bandpass_coefficient(
                bandpass, freq, output_unit
            )
            bandpass_scaling = self._get_bandpass_scaling(
                freq, bandpass, spectral_parameters
            )

            if self.diffuse:
                emission = amp*bandpass_scaling*bandpass_coefficient
                if fwhm.value != 0.0:
                    if self.is_polarized:
                        emission = hp.smoothing(
                            emission, fwhm=fwhm.to(u.rad).value
                        )*emission.unit
                    else:
                        emission[0] = hp.smoothing(
                            emission[0], fwhm=fwhm.to(u.rad).value
                        )*emission.unit   

            else:
                emission = self.get_map(
                    amp=amp*bandpass_scaling, fwhm=fwhm
                )*bandpass_coefficient

            return emission.to(_get_astropy_unit(output_unit))


    def _get_bandpass_scaling(self, freqs, bandpass, spectral_parameters):
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
        interp_parameters = get_interp_parameters(spectral_parameters)

        # Component does not have any spatially varying spectral parameters
        if not interp_parameters:
            freq_scaling = self.get_freq_scaling(
                freqs, self.freq_ref, **spectral_parameters
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
                spectral_parameters.copy()
            )

        # Component has two sptatially varying spectral parameter
        elif len(interp_parameters) == 2:    
            return interp2d(
                self, freqs, bandpass, interp_parameters, 
                spectral_parameters.copy()
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
        if not self.diffuse:
            return

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



class Synchrotron(Component):
    r"""Synchrotron component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.1 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{s}}(\nu) \propto
        \left( \frac{\nu}{\nu_\mathrm{0,s}} \right)^
        {\beta + C \ln \nu / \nu_{0,s}}.

    This is a generic power law given at a reference frequency :math:`\nu_{s,0}`
    with a power law :math:`\beta` in Rayleigh-Jeans temperature.
    :math:`C` is set to 0 for all current implementations as of BP9.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,s}` for the amplitude 
        template in units of GHz. Shape is either (1,) or (3, 1)
    beta : `numpy.ndarray`, `astropy.units.Quantity`
        The power law spectral index :math:`\beta`. The spectral index can 
        vary over the sky, and is therefore commonly given as a 
        shape (3, `npix`) array, but it can take the value of a scalar.

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Emission templates of synchrotron at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,s}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    get_freq_scaling    
    __call__
    """

    label = 'synch'
    diffuse = True


    def __init__(self, amp, freq_ref, beta):
        super().__init__(amp, freq_ref, beta=beta)


    def get_freq_scaling(self, freq, freq_ref, beta):
        r"""Computes the frequency scaling :math:`f_{\mathrm{s}}(\nu)` 
        from the reference frequency :math:`\nu_\mathrm{0,s}` to a frequency 
        :math:`\nu`, 

        .. math::

            f_{\mathrm{s}}(\nu) = \left( \frac{\nu}{\nu_\mathrm{0,s}} \right)
            ^{\beta}.

        Parameters
        ----------
        freq : `astropy.units.Quantity`)
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the amplitude map.
        beta : `numpy.ndarray`, `astropy.units.Quantity`
            The power law spectral index.
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.

        """
        scaling = (freq/freq_ref)**beta
        return scaling



class Dust(Component):
    r"""Thermal dust component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.3 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{d}}(\nu) \propto 
        \frac{\nu^{\beta_{\mathrm{d}}+1}}{\mathrm{e}^{h\nu/kT_{\mathrm{d}}}-1}.

    This is a modified blackbody with a power law spectral index :math:`\beta` 
    in Rayleigh-Jeans temperature, and thermal dust temperature 
    :math:`T_{\mathrm{d}}`.
    

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of the component at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies for the amplitude 
        template in units of GHz. Shape is either (1,) or (3, 1)
    beta : `numpy.ndarray`, `astropy.units.Quantity`
        The power law spectral index :math:`\beta`. The spectral index can vary 
        over the sky, and is therefore commonly given as a shape (3, `npix`) 
        array, but it can take the value of a scalar.
    T : `astropy.units.Quantity`:
        Temperature of the blackbody with unit :math:`\mathrm{K}_\mathrm{RJ}`.
        Can be a single value or a map with shape (`npix`,).

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Emission templates of synchrotron at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,d}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    get_freq_scaling    
    __call__
    """

    label = 'dust'
    diffuse = True


    def __init__(self, amp, freq_ref, beta, T):
        super().__init__(amp, freq_ref, beta=beta, T=T)


    def get_freq_scaling(self, freq, freq_ref, beta, T):
        r"""Computes the frequency scaling :math:`f_{\mathrm{d}}(\nu)` from the 
        reference frequency :math:`\nu_\mathrm{0,d}` to a frequency 
        :math:`\nu`, given the spectral index :math:`\beta` and the 
        electron temperature :math:`T_\mathrm{d}`.

        .. math::

            f_{\mathrm{d}}(\nu) = \left( \frac{\nu}{\nu_\mathrm{0,d}} \right)
            ^{\beta-2}\frac{B_\nu(T_{\mathrm{d}})}
            {B_{\nu_\mathrm{0,d}}(T_{\mathrm{d}})},

        where :math:`B_\nu(T_\mathrm{d})` is the blackbody emission.

        Parameters
        ----------
        freq : `astropy.units.Quantity`)
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the amplitude map.
        beta : `numpy.ndarray`, `astropy.units.Quantity`
            The power law spectral index.
        T : `astropy.units.Quantity`
            Temperature of the blackbody.
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

        blackbody_ratio = (
            blackbody_emission(freq, T) / blackbody_emission(freq_ref, T)
        )
        scaling = (freq/freq_ref)**(beta+1) * blackbody_ratio
        return scaling



class FreeFree(Component):
    r"""Free-free component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.2 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{ff}}(\nu) \propto
        \frac{g_{\mathrm{ff}}(T_\mathrm{e})}{\nu^2},


    where :math:`g_\mathrm{ff}` is the Gaunt factor, and :math:`T_\mathrm{e}` 
    is the electron temperature.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,ff}` for the amplitude 
        template in units of GHz. Shape is either (1,) or (3, 1)
    Te : `astropy.units.Quantity`
        Electron temperature map with unit K.

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Emission templates of synchrotron at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,\mathrm{ff}}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    get_freq_scaling    
    __call__
    """

    label = 'ff'
    diffuse = True


    def __init__(self, amp, freq_ref, Te):
        super().__init__(amp, freq_ref, Te=Te)


    def get_freq_scaling(self, freq, freq_ref, Te):
        r"""Computes the frequency scaling :math:`f_{\mathrm{ff}}(\nu)` from the 
        reference frequency :math:`\nu_{0, \mathrm{ff}}` to a frequency 
        :math:`\nu`, given the electron temperature :math:`T_\mathrm{e}`.

        .. math::

            f_{\mathrm{ff}}(\nu) = \frac{g_{\mathrm{ff}}\left(\nu ; T_{e}\right)}
            {g_{\mathrm{ff}}\left(\nu_{0, \mathrm{ff}} ; T_{e}\right)}
            \left(\frac{\nu_{0, \mathrm{ff}}}{\nu}\right)^{2}.

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the amplitude map.
        Te : `astropy.units.Quantity`
            Electron temperature.
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

        gaunt_factor_ratio = gaunt_factor(freq, Te) / gaunt_factor(freq_ref, Te)
        scaling = (freq_ref/freq)**2 * gaunt_factor_ratio
        return scaling



class AME(Component):
    r"""Spinning dust component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.4 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::


        \mathbf{s}_{\mathrm{RJ}}^{\mathrm{sd}}(\nu) \propto 
        \nu^{-2} s_{0}^{\mathrm{sd}}\left(\nu \cdot 
        \frac{30.0 \mathrm{GHz}}{\nu_{p}}\right)


    where the peak frequency :math:`\nu_p` is 30 GHz.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of the component at the reference frequencies given
        by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,sd}` for the amplitude 
        template in units of GHz. Shape is either (1,) or (3, 1)
    nu_p : `astropy.units.Quantity`
        Peak frequency.

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Emission templates of synchrotron at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,sd}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    get_freq_scaling    
    __call__
    """

    label = 'ame'
    diffuse = True

    def __init__(self, amp, freq_ref, nu_p):
        super().__init__(amp, freq_ref, nu_p=nu_p)

        # Read in spdust2 template
        SPDUST2_FILE = DATA_DIR / 'spdust2_cnm.dat'
        spdust2_freq, spdust2_amp = np.loadtxt(SPDUST2_FILE, unpack=True)
        spdust2_freq = u.Quantity(spdust2_freq, unit=u.GHz)
        spdust2_amp = u.Quantity(spdust2_amp, unit=(u.Jy/u.sr)).to(
            u.K, equivalencies=u.brightness_temperature(spdust2_freq)
        )        
        self.spdust2 = np.array([spdust2_freq.si.value, spdust2_amp.si.value])


    def get_freq_scaling(self, freq, freq_ref, nu_p):
        r"""Computes the frequency scaling :math:`f_{\mathrm{sd}}(\nu)` from the 
        reference frequency :math:`\nu_{0, \mathrm{sd}}` to a frequency 
        :math:`\nu`, given the peak frequency :math:`\nu_p`.

        .. math::

            f_{\mathrm{sd}}(\nu) = \left(\frac{\nu_{0, \mathrm{sd}}}{\nu}\right)^{2} 
            \frac{s_{0}^{\mathrm{sd}}\left(\nu \cdot 
            \frac{\nu_{p}}{30.0 \mathrm{GHz}}\right)}
            {s_{0}^{\mathrm{sd}}\left(\nu_{0, \mathrm{sd}} \cdot \frac{\nu_{p}}
            {30.0 \mathrm{GHz}}\right)}  


        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the amplitude map.
        nu_p : `astropy.units.Quantity`
            Peak frequency.
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

        spdust2 = self.spdust2
        peak_scale = 30*u.GHz / nu_p

        # AME is undefined at outside of this frequency range
        if not (np.min(spdust2[0]) < ((freq*peak_scale).si.value).any() < np.max(spdust2[0])):
            return u.Quantity(0, unit=u.dimensionless_unscaled)

        interp = np.interp((freq*peak_scale).si.value, spdust2[0], spdust2[1])
        interp_ref = (
            np.interp((freq_ref*peak_scale).si.value, spdust2[0], spdust2[1])
        )
        scaling = interp/interp_ref
        return scaling



class CMB(Component):
    r"""CMB component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.2 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \mathbf{s}_{\mathrm{RJ}}^{\mathrm{CMB}}(\nu) \propto \frac{x^{2} 
        \mathrm{e}^{x}}{\left(\mathrm{e}^{x}-1\right)^{2}} 
        \boldsymbol{s}^{\mathrm{CMB}}


    where :math:`x=h v / k T_{0}` and :math:`T_0 = 2.7255 \mathrm{K}` as of BP9.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of the component at the reference frequencies given
        by freq_ref.

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Emission templates of CMB in units of :math:`\mathrm{K}_{\mathrm{CMB}}´
    freq_ref : `astropy.units.Quantity`
        Reference frequency for CMB is set to ``None``.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    remove_dipole
    get_freq_scaling    
    __call__
    """

    label = 'cmb'
    diffuse = True

    def __init__(self, amp):
        super().__init__(amp, freq_ref=None)


    def remove_dipole(self, return_dipole=False, gal_cut=10):
        """Removes the solar dipole from the reference amplitude map.

        Parameters
        ----------
        return_dipole : bool
            If ``True``, a map of the dipole is returned. Defaut: ``False``.
        gal_cut : float
            Galactic latitude coordinate. Default: 10 degrees.

        Returns
        -------
        dipole : `astropy.units.Quantity`
            If `return_dipole` is ``True``, return the dipole map.
        """

        if not return_dipole:
            hp.remove_dipole(self.amp[0], gal_cut=gal_cut, copy=False)
        else: 
            amp_without_dipole = u.Quantity(
                hp.remove_dipole(
                    self.amp[0], gal_cut=gal_cut
                ), unit=self.amp.unit
            )
            dipole = self.amp[0] - amp_without_dipole
            self.amp[0] = amp_without_dipole

            return dipole


    def get_freq_scaling(self, freq, freq_ref=None):
        r"""Computes the frequency scaling factor :math:`f_{\mathrm{CMB}}(\nu)`. 
        For the CMB component, the frequency scaling factor is given by the 
        unit conversion factor from :math:`\mathrm{K}_\mathrm{CMB}` to 
        :math:`\mathrm{K}_\mathrm{RJ}` since the amplitude template is in 
        units of :math:`\mathrm{K}_\mathrm{CMB}`

        .. math::

            f_{\mathrm{CMB}}(\nu) = \frac{x^{2} e^{x}}{\left(e^{x}-1\right)^{2}}
    

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the amplitude map. Default: ``None``
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

        return thermodynamical_to_brightness(freq)



class Radio(Component):
    r"""Point source component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.4.1 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \mathbf{s}_{\mathrm{RJ}}^{\mathrm{src}}(\nu) \propto
        \left(\frac{\nu}{\nu_{\mathrm{0, src}}}\right)^{\alpha-2}


    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Sampled amplitudes for each point source.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,src}` for the point source 
        amplitudes.
    specind : `astropy.units.Quantity`
        Power law spectral index :math:`\alpha`.

    Attributes
    ----------
    label : str
        Component label.
    diffuse : bool
        Whether or not the component is diffuse in nature.
    amp : `astropy.units.Quantity`
        Point source amplitudes at the reference frequencies given
        by `freq_ref`. Note that this is not a healpix map.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,\mathrm{src}}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters.

    Methods
    -------
    get_freq_scaling
    get_map
    __call__
    """

    label = 'radio'
    diffuse = False


    def __init__(self, amp, freq_ref, specind):
        super().__init__(amp, freq_ref, specind=specind)
        self.amp = u.Quantity(self.amp.value, unit='mJy')
        self.angular_coords = self._read_angular_coords()
        self.spectral_parameters['specind'] = np.squeeze(
            self.spectral_parameters['specind'][0]
        )


    def _set_nside(self, nside):
        """Imports the nside of the sky model."""
        self.nside = nside


    def _read_angular_coords(self, catalog=DATA_DIR/'radio_catalog.dat'):
        """Reads in the angular coordinates of the point sources from a given 
        catalog.

        TODO: Make sure that the correct catalog is selected for a given chain
        (in case catalogs change from run to run)

        Parameters:
        -----------
        catalog: str
            Path to the point source catalog. Default is the COM_GB6 catalog.
        
        Returns:
        --------
        coords : `numpy.ndarray`
            Longitude and latitude values of each point source. Shape is 
            (2,n_pointsources).

        """
        try:
            coords = np.loadtxt(catalog, usecols=(0,1))
        except OSError:
            raise OSError('Could not find point source catalog')

        if len(coords) == len(self.amp[0]):
            return coords
        else:
            raise ValueError('Cataloge does not match chain catalog')
        

    @u.quantity_input(amp=u.Jy, fwhm=(u.rad, u.arcmin, u.deg))
    def get_map(self, amp, nside='model_nside', fwhm=0.0*u.rad, 
                sigma=None, n_fwhm=2):
        """Maps the cataloged radio source points onto a healpix map with a 
        truncated gaussian beam. For more information, see 
        `Mitra et al. <https://arxiv.org/pdf/1005.1929.pdf>`_.

        Parameters
        ----------
        amp : `astropy.units.Quantity`
            Amplitude of the radio sources.
        nside : int
            The nside of the output map. If component is part of a sky model, 
            we automatically select the model nside. Must be >= 32 to not throw
            exception. Default: 'model_nside'.
        fwhm : `astropy.units.Quantity`
            The full width half max parameter of the Gaussian. Default: 0.0
        sigma : float
            The sigma of the Gaussian (beam radius). Overrides fwhm. 
            Default: None
        n_fwhm : int, float
            The fwhm multiplier used in computing radial cut off r_max 
            calculated as r_max = n_fwhm * fwhm of the Gaussian.
            Default: 2.
        """

        if nside == 'model_nside':
            try:
                nside = self.nside
            # Can occur when the radio component is used outside of a sky model
            except AttributeError:
                raise AttributeError(
                    'Component is not part of a sky model. Please provide an explicit nside'
                )
        
        amp = np.squeeze(amp)
        radio_template = u.Quantity(
            np.zeros(hp.nside2npix(nside)), unit=amp.unit
        )
        pix_lon, pix_lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True
        )
        # Point source coordinates
        angular_coords = self.angular_coords

        fwhm = fwhm.to(u.rad)
        if sigma is None:
            sigma = fwhm / (2*np.sqrt(2 * np.log(2)))

        # No smoothing nesecarry. Directly map sources to pixels
        if sigma == 0.0:
            warnings.warn('mapping point sources to pixels without beam smoothing.')
            pixels = hp.ang2pix(nside, *angular_coords.T, lonlat=True)
            beam_area = hp.nside2pixarea(nside)*u.sr
            radio_template[pixels] = amp

        # Apply gaussian (or other beam) to point source and neighboring pixels
        else:
            pix_res = hp.nside2resol(nside)
            if fwhm.value < pix_res:
                raise ValueError(
                    'fwhm must be >= pixel resolution to resolve the '
                    'point sources.'
                )
            beam_area = 2 * np.pi * sigma**2
            r_max = n_fwhm * fwhm.value

            with tqdm(total=len(angular_coords), file=sys.stdout) as pbar:
                sigma = sigma.value
                print('Smoothing point sources')

                for idx, (lon, lat) in enumerate(angular_coords):
                    vec = hp.ang2vec(lon, lat, lonlat=True)
                    inds = hp.query_disc(nside, vec, r_max)
                    r = hp.rotator.angdist(
                        np.array(
                            [pix_lon[inds], pix_lat[inds]]), np.array([lon, lat]
                        ), lonlat=True
                    )
                    radio_template[inds] += amp[idx]*gaussian_beam_2D(r, sigma)
                    pbar.update()

        radio_template = radio_template.to(
            u.uK, u.brightness_temperature(self.freq_ref, beam_area)
        )

        return np.expand_dims(radio_template, axis=0)


    def get_freq_scaling(self, freq, freq_ref, specind):
        r"""Computes the frequency scaling :math:`f_{\mathrm{src}}(\nu)` 
        from the reference frequency :math:`\nu_{0, \mathrm{src}}` to a 
        frequency :math:`\nu`, given the spectral index :math:`\alpha`.

        .. math::

            f_{\mathrm{src}}(\nu) = \sum_{j=1}^{N_{\mathrm{src}}} 
            \left(\frac{\nu}{\nu_{0, \mathrm{src}}}\right)
            ^{\alpha_{j, \mathrm{src}}-2}

        where we sum over each point source :math:`j`.

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the point source amplitudes.
        specind : `astropy.units.Quantity`
            Power law spectral index :math:`\alpha`.
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.

        """

        scaling = (freq/freq_ref)**(specind-2)
        return scaling
