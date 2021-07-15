from cosmoglobe.sky.templates import (
    DiffuseComponent,
    PointSourceComponent,
    LineComponent,
)
from cosmoglobe.utils.functions import (
    blackbody_emission,
    gaunt_factor, 
    thermodynamical_to_brightness
)

from pathlib import Path
from sys import exit
import warnings
import astropy.units as u
import numpy as np
import healpy as hp

warnings.simplefilter('once', UserWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RADIO_CATALOG = DATA_DIR / 'radio_catalog.dat'
SPDUST2_FILE = DATA_DIR / 'spdust2_cnm.dat'



class Synchrotron(DiffuseComponent):
    r"""Class representing the synchrotron component in the Cosmoglobe 
    Sky Model.

    Notes
    -----
    This is a generic power law given at a reference frequency 
    :math:`\nu_{s,0}` with a power law :math:`\beta` in Rayleigh-Jeans 
    temperature. It is defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.1 
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{s}}(\nu) \propto
        \left( \frac{\nu}{\nu_\mathrm{0,s}} \right)^
        {\beta + C \ln \nu / \nu_{0,s}}.


    :math:`C` is set to 0 for all current implementations as of BP9.

    Attributes
    ----------
    amp : `astropy.units.Quantity`
        Emission templates of synchrotron at the reference frequencies given
        by `freq_ref`.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies :math:`\nu_\mathrm{0,s}` for the amplitude 
        template in units of GHz.
    spectral_parameters : dict
        Dictionary containing the spectral parameters :math:`\beta` and 
        :math:`T`.
    label : str
        Component label.

    Methods
    -------
    get_freq_scaling    
    __call__
    """

    label = 'synch'

    def __init__(self, amp, freq_ref, beta):
        """
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
        """

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



class Dust(DiffuseComponent):
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
    comp_type : str
        Component type. Must be either 'diffuse', 'line', or 'ptsrc'. 
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
        scaling = (freq/freq_ref)**(beta-2) * blackbody_ratio

        return scaling



class FreeFree(DiffuseComponent):
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
    comp_type : str
        Component type. Must be either 'diffuse', 'line', or 'ptsrc'. 
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



class AME(DiffuseComponent):
    r"""Spinning dust component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.3.4 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::


        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{sd}}(\nu) \propto 
        \nu^{-2} \boldsymbol{s}_{0}^{\mathrm{sd}}\left(\nu \cdot 
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
    comp_type : str
        Component type. Must be either 'diffuse', 'line', or 'ptsrc'. 
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

    def __init__(self, amp, freq_ref, nu_p):
        super().__init__(amp, freq_ref, nu_p=nu_p)

        # Read in spdust2 template
        spdust2_freq, spdust2_amp = np.loadtxt(SPDUST2_FILE, unpack=True)
        spdust2_freq = u.Quantity(spdust2_freq, unit=u.GHz)
        spdust2_amp = u.Quantity(
            spdust2_amp, unit=(u.Jy/u.sr)
        )
        spdust2_amp = spdust2_amp.to(
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
        if not (
            np.min(spdust2[0]) < (freq*peak_scale).si.value.any() < np.max(spdust2[0])
        ):
            return u.Quantity(0, unit=u.dimensionless_unscaled)

        interp = np.interp((freq * peak_scale).si.value, spdust2[0], spdust2[1])
        interp_ref = (
            np.interp((freq_ref * peak_scale).si.value, spdust2[0], spdust2[1])
        )
        scaling = interp / interp_ref

        return scaling



class CMB(DiffuseComponent):
    r"""CMB component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.2 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{CMB}}(\nu) \propto \frac{x^{2} 
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
    comp_type : str
        Component type. Must be either 'diffuse', 'line', or 'ptsrc'. 
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

    def __init__(self, amp, freq_ref=None):
        super().__init__(amp, freq_ref=freq_ref)


    def remove_dipole(self, return_dipole=False, gal_cut=10):
        """Removes the solar dipole from the reference amplitude map.

        Parameters
        ----------
        return_dipole : bool
            If ``True``, a map of the dipole is returned. Defaut: ``False``.
        gal_cut : float
            Masks pixles :math:`\pm` `gal_cut` in latitude before estimating 
            dipole. Default: 10 degrees.

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



class Radio(PointSourceComponent):
    r"""Point source component class. Defined using the convention in 
    `BeyondPlanck (2020), Section 3.4.1 <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{src}}(\nu) \propto
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
    comp_type : str
        Component type. Must be either 'diffuse', 'line', or 'ptsrc'. 
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

    def __init__(self, amp, freq_ref, specind):
        super().__init__(amp, freq_ref, specind=specind)

        self.amp = u.Quantity(self.amp.value, unit='mJy')
        self.angular_coords = self._read_coords(RADIO_CATALOG)
        self.spectral_parameters['specind'] = np.squeeze(
            self.spectral_parameters['specind'][0]
        )


    def _read_coords(self, catalog):
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