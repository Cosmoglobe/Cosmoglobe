from ..tools.map import to_stokes
from ..tools.bandpass import _extract_scalars
from ..science.functions import blackbody_emission

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d, RectBivariateSpline

class Component:
    """Base class for all sky components.

    Any sky component you make should subclass this class. All components must 
    implement the get_freq_scaling method. This method needs to return the 
    frequency scaling factor from the reference frequency to a arbritrary 
    frequency, as a function of some spectral parameters. Following is an 
    example of a custom implementation of a component whos emission scales as 
    a power law:

        import cosmoglobe.sky as sky

        class PowerLaw(sky.Component):
            def __init__(self, comp_name, amp, freq_ref, beta):
                super().__init__(comp_name, amp, freq_ref, beta=beta)

            def get_freq_scaling(self, freq, beta):
                return (freq/freq_ref)**beta

    Args:
    -----
    comp_name (str):
        Name/label of the component, e.g 'dust'. Is used to set the component
        attribute in a cosmoglobe.Model.
    amp (np.ndarray, astropy.units.quantity.Quantity, cosmoglobe.IQUMap):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (astropy.units.quantity.Quantity):
        Reference frequencies for the amplitude map in units of Hertz.
    spectrals (dict):
        Spectral parameters required to compute the frequency scaling factor. 
        can be scalars, numpy arrays or astropy quantities. Default: None

    """
    def __init__(self, comp_name, amp, freq_ref, **spectrals):
        self.comp_name = comp_name
        self.amp = to_stokes(amp, freq_ref=freq_ref, label=self.comp_name)

        self._spectrals = {}
        if spectrals is not None:
            for key, value in spectrals.items():
                self._spectrals[key] = value
                setattr(self, key, self._spectrals[key])


    @u.quantity_input(freq=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, freq, bandpass=None, output_unit=None):
        """Returns the component emission at an arbitrary frequency.

        TODO: Implement bandpass normalization and output_unit

        Args:
        -----
        freq (astropy.units.quantity.Quantity):
            A frequency, or a list of frequencies at which to evaluate the 
            component emission.
        bandpass (astropy.units.quantity.Quantity):
            Bandpass profile in units of K_RJ or Jy/sr corresponding to the
            frequency list (freq). If None, a delta peak in frequency is 
            assumed. Default : None
        output_unit (astropy.units.Unit):
            The desired output units of the emission. Must be signal units, e.g 
            Jy/sr or K. Default: None

        Returns
        -------
        emission (astropy.units.quantity.Quantity):
            Model emission at the given frequency.

        """
        # Convert all values to si and prepare broadcasting
        freq = freq.si
        freq_ref = np.expand_dims(self.amp.freq_ref.si, axis=1)
        spectrals = {key:(value.si if isinstance(value, u.Quantity) else value)
                     for key, value in self._spectrals.items()}

        if bandpass is not None:
            scaling = self._get_bandpass_conversion(freq, freq_ref, bandpass)

        else:
            if freq.ndim > 0:
                scalings = []
                for freq in freq:
                    scaling = self.get_freq_scaling(freq, freq_ref, **spectrals)
                    scalings.append(scaling)
                scaling = u.Quantity(scalings)
            else:
                scaling = self.get_freq_scaling(freq, freq_ref, **spectrals)

        return self.amp*scaling


    def _get_bandpass_conversion(self, freqs, freq_ref, bandpass_array, n=10):
        """Returns the frequency scaling factor given a frequency array and a 
        bandpass profile.

        Makes use of the mixing matrix implementation from Commander3. For 
        more information, see section 4.2 in https://arxiv.org/abs/2011.05609.

        TODO: FIX 2D interp case.
        TODO: Test that this actually works for real components. Use the fact 
        that.
        TODO: Rewrite in a more functional and general way.
        dicts are ordered in >3.6

        Args:
        -----
        freqs : astropy.units.quantity.Quantity
            Frequencies corresponding to the bandpass weights.
        freq_ref : tuple, list, np.ndarray
            Reference frequencies for the amplitude map.
        bandpass_array : astropy.units.quantity.Quantity
            Normalized bandpass profile. Must have signal units.
        n : int
            number of values to interpolate over.
            Default : 10

        Returns:
        --------
        float, np.ndarray
            Frequency scaling factor given a bandpass.

        """
        if self.amp._has_pol:
            component_is_polarized = True
        else:
            component_is_polarized = False

        scalars = _extract_scalars(self._spectrals)
        interp_ranges = {}

        for key, value in self._spectrals.items():
            if scalars is None or key not in scalars:
                interp_ranges[key] = np.linspace(np.amin(value), 
                                                 np.amax(value), 
                                                 n) * value.unit

        # All spectral parameters are scalars. No need to interpolate
        if not interp_ranges:
            freq_scaling = self.get_freq_scaling(freqs, freq_ref, **scalars)
            integral = np.trapz(freq_scaling*bandpass_array, freqs)
            if self.amp._has_pol:
                integral = np.expand_dims(integral, axis=1)            
            return integral

        # Interpolating over one spectral parameter
        elif len(interp_ranges) == 1:
            integrals = []
            for key, value in interp_ranges.items():
                for spec in value:
                    spectrals = {key:spec}
                    if scalars is not None:
                        spectrals.update(scalars)
                    freq_scaling = self.get_freq_scaling(freqs, freq_ref, 
                                                         **spectrals)
                    integrals.append(
                        np.trapz(freq_scaling*bandpass_array, freqs)
                    )

                interp_range = interp_ranges[key]
                spectral_parameter = self._spectrals[key]
                if component_is_polarized:
                    # If component is polarized, converts list to shape (3, n)
                    integrals = np.transpose(integrals)
                    conversion_factor = []
                    for i in range(3):
                        f = interp1d(interp_range, integrals[i])
                        if spectral_parameter.ndim > 1:
                            conversion_factor.append(f(spectral_parameter[i]))
                        else:
                            conversion_factor.append(f(spectral_parameter)[0])
                    return conversion_factor
                
                # print(integrals)
                print(np.array(integrals))
                f = interp1d(interp_range, integrals)
                return f(spectral_parameter)

        # Interpolating over two spectral parameter
        elif len(interp_ranges) == 2:
            spectrals_keys = list(interp_ranges.keys())
            spectrals_values = list(interp_ranges.values())
            meshgrid = np.meshgrid(*spectrals_values)

            # Converting np.zeros array to correct units
            integral_unit = (bandpass_array.unit * freqs.unit)
            if component_is_polarized:
                integrals = np.zeros((n, n, 3)) * integral_unit
            else:
                integrals = np.zeros((n, n)) * integral_unit
                
            for i in range(n):
                for j in range(n):
                    # Unpacking a dictionary makes sure that the spectral
                    # values are mapped to the correct parameters. 
                    spectrals = {
                        spectrals_keys[0]: meshgrid[0][i,j],
                        spectrals_keys[1]: meshgrid[1][i,j]
                    }
                    freq_scaling = self.get_freq_scaling(freqs, freq_ref, 
                                                         **spectrals)
                    integral = np.trapz(freq_scaling*bandpass_array, freqs)
                    integrals[i,j] = integral

            if component_is_polarized:
                integrals = np.transpose(integrals)
                conversion_factor = []
                spectral_values = list(self._spectrals.values())
                for i in range(3):
                    f = RectBivariateSpline(*spectrals_values, integrals[i])
                    # conversion_factor.append(f(list(zip(*spectral_values))[i], grid=False))
                return conversion_factor

            f = RectBivariateSpline(*spectrals_values, integrals)
            return f(*list(self._spectrals.values()), grid=False)

        else:
            return NotImplemented


    def __repr__(self):
        main_repr = f'{self.__class__.__name__}'
        main_repr += '(amp, freq_ref, '
        for spectral in self._spectrals:
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
        in a cosmoglobe.Model.
    amp (np.ndarray, astropy.units.quantity.Quantity, cosmoglobe.IQUMap):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (astropy.units.quantity.Quantity):
        Reference frequencies for the amplitude map in units of Hertz.
    beta (np.ndarray, astropy.units.quantity.Quantity):
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
        freq (astropy.units.quantity.Quantity):
            Frequency at which to evaluate the model.
        freq_ref (astropy.units.quantity.Quantity):
            Reference frequencies for the amplitude map.
        beta (numpy.ndarray, astropy.units.quantity.Quantity):
            The power law spectral index.
            
        Returns:
        --------
        scaling (astropy.units.quantity.Quantity):
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
        in a cosmoglobe.Model.
    amp (np.ndarray, astropy.units.quantity.Quantity, cosmoglobe.IQUMap):
        Amplitude templates at the reference frequencies for I or IQU stokes 
        parameters.
    freq_ref (astropy.units.quantity.Quantity):
        Reference frequencies for the amplitude map in units of Hertz.
    beta (np.ndarray, astropy.units.quantity.Quantity):
        The power law spectral index. The spectral index can vary over the sky, 
        and is therefore commonly given as a shape (3, nside) array, but it can 
        take the value of a scalar.
    T (astropy.units.quantity.Quantity):
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
        freq (astropy.units.quantity.Quantity):
            Frequency at which to evaluate the model.
        freq_ref (astropy.units.quantity.Quantity):
            Reference frequencies for the amplitude map.
        beta (numpy.ndarray, astropy.units.quantity.Quantity):
            The power law spectral index.
        T (astropy.units.quantity.Quantity): 
            Temperature of the blackbody.
            
        Returns:
        --------
        scaling (astropy.units.quantity.Quantity):
            Frequency scaling factor with dimensionless units.

        """

        scaling = (freq/freq_ref)**(beta-2)
        scaling *= blackbody_emission(freq, T) / blackbody_emission(freq_ref, T)

        return scaling