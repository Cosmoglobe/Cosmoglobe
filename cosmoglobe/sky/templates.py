from cosmoglobe.utils.bandpass import (
    get_bandpass_coefficient,
    get_interp_parameters, 
    get_normalized_bandpass, 
    interp1d,
    interp2d,
)
from cosmoglobe.utils.utils import (
    _get_astropy_unit, 
    emission_to_unit,
    gaussian_beam_2D
)

from tqdm import tqdm
from sys import exit
import warnings
import astropy.units as u
import numpy as np
import healpy as hp
import sys


class Component:
    """Base class for a sky component from the Cosmoglobe Sky Model.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates or lists of point source amplitudes of the 
        component at the reference frequencies given by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies for the amplitude template in units of Hertz.
    spectral_parameters : dict
        Spectral parameters required to compute the frequency scaling factor. 
    """

    label = None

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
        elif freq_ref.size == 1:
            return freq_ref
        elif freq_ref.size == 2:
            return np.expand_dims(
                u.Quantity([freq_ref[0], freq_ref[1], freq_ref[1]]), axis=1
            )
        elif freq_ref.size == 3:
            return freq_ref.reshape((3,1))
        else:
            raise ValueError('Unrecognized shape.')


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None), 
                      fwhm=(u.rad, u.deg, u.arcmin))
    def __call__(self, freqs, bandpass=None, fwhm=0.0*u.rad, output_unit=u.uK):
        r"""Computes the simulated component emission at an arbitrary frequency 
        or integrated over a bandpass.

        .. math::

            \boldsymbol{s}_\mathrm{comp} = \boldsymbol{a}_\mathrm{comp} \; 
            f_{\mathrm{comp}}(\nu)

        where :math:`\boldsymbol{a}_\mathrm{comp}` is the amplitude template of 
        the component at some reference frequency, and 
        :math:`f_\mathrm{comp}(\nu)` is the scaling factor for 
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
        # Assume delta frequency
        if freqs.size == 1:
            emission = self.get_delta_emission(
                freqs, fwhm=fwhm, output_unit=output_unit
            )

        else:
            emission = self.get_bandpass_emission(
                freqs, bandpass, fwhm=fwhm, output_unit=output_unit
            )

        if fwhm != 0.0 and not isinstance(self, PointSourceComponent):
            if self.is_polarized:
                emission = u.Quantity(
                    hp.smoothing(emission, fwhm=fwhm.to(u.rad).value),
                    unit=emission.unit
                )
            else:
                emission[0] = u.Quantity(
                    hp.smoothing(emission[0], fwhm=fwhm.to(u.rad).value),
                    unit=emission.unit
                )

        return emission


    def _get_bandpass_scaling(self, freqs, bandpass, spectral_parameters):
        """Returns the frequency scaling factor given a bandpass profile and a
        corresponding frequency array.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            Bandpass profile frequencies.
        bandpass : `astropy.units.Quantity`
            Normalized bandpass profile.

        Returns
        -------
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
        if self.comp_type == 'ptsrc':
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



class DiffuseComponent(Component):
    """Class for a diffuse sky component.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates or lists of point source amplitudes of the 
        component at the reference frequencies given by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies for the amplitude template in units of Hertz.
    spectral_parameters : dict
        Spectral parameters required to compute the frequency scaling factor. 
    """

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        # Expand dimension on rank-1 arrays from from (n,) to (n, 1) to support
        # broadcasting with (1, nside) or (3, nside) arrays
        self.amp = amp if amp.ndim != 1 else np.expand_dims(amp, axis=0)
        self.spectral_parameters = {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in spectral_parameters.items()
        }

    
    def get_delta_emission(self, freq, fwhm=0.0*u.rad, output_unit=u.uK):
        """Computes the simulated component emission at a delta frequency

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            A frequency.
        output_unit : `astropy.units.Unit`
            The desired output unit of the emission. Must be signal units. 
            Default: None

        Returns
        -------
        `astropy.units.Quantity`
            Delta frequency emission.
        """

        scaling = self.get_freq_scaling(
            freq, self.freq_ref, **self.spectral_parameters
        )
        emission = self.amp * scaling

        if output_unit is not None:
            emission = emission_to_unit(emission, freq, output_unit)

        return emission

    
    def get_bandpass_emission(self, freqs, bandpass=None, fwhm=0.0*u.rad, output_unit=u.uK):
        """Computes the simulated component emission integrated over a bandpass.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            Bandpass frequencies.
        bandpass : `astropy.units.Quantity`
            Bandpass profile in signal units. The shape of the bandpass must
            match that of the freq input. Default : None
        output_unit : `astropy.units.Unit`
            The desired output unit of the emission. Must be signal units. 
            Default: None

        Returns
        -------
        `astropy.units.Quantity`
            Emission.
        """

        # create top hat weights
        if bandpass is None:
            warnings.warn('No bandpass was passed. Default to top-hat bandpass')
            bandpass = np.ones(len(freqs))/len(freqs) * u.K

        # Perform bandpass integration
        bandpass = get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = get_bandpass_coefficient(
            bandpass, freqs, output_unit
        )

        bandpass_scaling = self._get_bandpass_scaling(
            freqs, bandpass, self.spectral_parameters
        )

        emission = self.amp * bandpass_scaling * bandpass_coefficient

        return emission.to(_get_astropy_unit(output_unit))




class PointSourceComponent(Component):
    """Class for a point source sky component.

    Parameters
    ----------
    amp : `astropy.units.Quantity`
        Emission templates or lists of point source amplitudes of the 
        component at the reference frequencies given by freq_ref.
    freq_ref : `astropy.units.Quantity`
        Reference frequencies for the amplitude template in units of Hertz.
    spectral_parameters : dict
        Spectral parameters required to compute the frequency scaling factor. 
    """

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        # Expand dimension on rank-1 arrays from from (n,) to (n, 1) to support
        # broadcasting with (1, nside) or (3, nside) arrays
        self.spectral_parameters = {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in spectral_parameters.items()
        }

    def get_delta_emission(self, freq, fwhm=0.0*u.rad, output_unit=u.uK):
        """Computes the simulated component emission at a delta frequency

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            A frequency.
        output_unit : `astropy.units.Unit`
            The desired output unit of the emission. Must be signal units. 
            Default: None

        Returns
        -------
        `astropy.units.Quantity`
            Delta frequency emission.
        """

        scaling = self.get_freq_scaling(
            freq, self.freq_ref, **self.spectral_parameters
        )

        scaled_amps = self.amp*scaling
        emission = self.points_to_map(scaled_amps, fwhm=fwhm)

        if output_unit is not None:
            emission = emission_to_unit(emission, freq, output_unit)

        return emission



    def get_bandpass_emission(self, freqs, bandpass=None, fwhm=0.0*u.rad, output_unit=u.uK):
        """Computes the simulated component emission integrated over a bandpass.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            Bandpass frequencies.
        bandpass : `astropy.units.Quantity`
            Bandpass profile in signal units. The shape of the bandpass must
            match that of the freq input. Default : None
        output_unit : `astropy.units.Unit`
            The desired output unit of the emission. Must be signal units. 
            Default: None

        Returns
        -------
        `astropy.units.Quantity`
            Emission.
        """

        if bandpass is None:
            warnings.warn('No bandpass was passed. Default to top-hat bandpass')
            bandpass = np.ones(len(freqs))/len(freqs) * u.K

        # Perform bandpass integration
        bandpass = get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = get_bandpass_coefficient(
            bandpass, freqs, output_unit
        )

        bandpass_scaling = self._get_bandpass_scaling(
            freqs, bandpass, self.spectral_parameters
        )

        scaled_amps = self.amp*bandpass_scaling
        emission = self.points_to_map(scaled_amps, fwhm=fwhm) * bandpass_coefficient

        return emission.to(_get_astropy_unit(output_unit))



    @u.quantity_input(amp=[u.Jy, u.K], fwhm=[u.rad, u.arcmin, u.deg])
    def points_to_map(self, amp, nside=None, fwhm=0.0*u.rad, sigma=None, n_fwhm=2):
        """Maps the cataloged point sources points onto a healpix map with a 
        truncated gaussian beam. For more information, see 
        `Mitra et al. (2010) <https://arxiv.org/pdf/1005.1929.pdf>`_.

        Parameters
        ----------
        amp : `astropy.units.Quantity`
            Amplitude of the point sources.
        nside : int
            The nside of the output map. If component is part of a sky model, 
            we automatically select the model nside. Must be >= 32 to not throw
            exception. Default: None.
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

        if nside is None:
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
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # No smoothing nesecarry. Directly map sources to pixels
        if sigma == 0.0:
            warnings.warn('mapping point sources to pixels without beam smoothing.')
            pixels = hp.ang2pix(nside, *angular_coords.T, lonlat=True)
            beam_area = hp.nside2pixarea(nside) * u.sr
            radio_template[pixels] = amp

        # Apply gaussian (or other beam) to point source and neighboring pixels
        else:
            pix_res = hp.nside2resol(nside)
            if fwhm.value < pix_res:
                raise ValueError(
                    'fwhm must be >= pixel resolution to resolve the '
                    'point sources.'
                )
            beam_area = 2 * np.pi * sigma ** 2
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
                    radio_template[inds] += amp[idx] * gaussian_beam_2D(r, sigma)
                    pbar.update()

        radio_template = radio_template.to(
            u.uK, u.brightness_temperature(self.freq_ref, beam_area)
        )

        return np.expand_dims(radio_template, axis=0)




class LineComponent(Component):

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)








