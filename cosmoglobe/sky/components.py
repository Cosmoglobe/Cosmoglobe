from ..tools.map import to_IQU
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
            def __init__(self, comp_name, amp, nu_ref, beta):
                super().__init__(comp_name, amp, nu_ref, beta=beta)

            def get_freq_scaling(self, nu, beta):
                return (nu / nu_ref) ** beta

    Args:
    -----
    comp_name : str
        Name/label of the component, e.g 'dust'. Is used to set the component
        attribute in a cosmoglobe.Model.
    amp : np.ndarray, astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        Amplitude map for I or IQU stokes parameters.
    nu_ref : astropy.units.quantity.Quantity
        Reference frequencies for the amplitude map. Each array element must 
        be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    spectrals: dict
        Spectral maps required to compute the frequency scaling factor. Maps 
        must be of types astropy.units.quantity.Quantity or cosmoglobe.IQUMap.
        Default : None

    """
    def __init__(self, comp_name, amp, nu_ref, **spectrals):
        self.comp_name = comp_name
        self.amp = to_IQU(amp, nu_ref=nu_ref, label=self.comp_name)

        self._spectrals = {}
        if spectrals is not None:
            for key, value in spectrals.items():
                if isinstance(value, np.ndarray):
                    self._spectrals[key] = to_IQU(
                        value, nu_ref=nu_ref, label=f'{self.comp_name} {key}'
                    )
                else:
                    self._spectrals[key] = value
                setattr(self, key, self._spectrals[key])


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=None):
        """Returns the full sky component emission at an arbitrary frequency nu.

        TODO: Implement bandpass normalization and output_unit

        Args:
        -----
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the 
            component emission.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile in units of K_RJ or Jy/sr corresponding to the
            frequency array, nu. If None, a delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.quantity.Quantity or str
            Desired unit for the output map. Must be a valid astropy.unit or 
            one of the two following strings 'K_CMB', 'K_RJ'.
            Default : None

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        nu = nu.si
        if self.amp._has_pol:   # Broadcasting compatibility
            nu_ref = np.expand_dims(self.amp.nu_ref.si, axis=1)
        else:
            nu_ref = self.amp.nu_ref.si

        if bandpass is None:
            if nu.ndim > 0:
                scalings = []
                for freq in nu:
                    scaling = self.get_freq_scaling(freq, nu_ref, **self._spectrals)
                    scalings.append(scaling)
                return self.amp*(scalings*u.dimensionless_unscaled)

            scaling = self.get_freq_scaling(nu, nu_ref, **self._spectrals)
            return self.amp*scaling

        scaling = self._get_bandpass_conversion(nu, nu_ref, bandpass)
        return self.amp*scaling


    def _get_bandpass_conversion(self, nu_array, nu_ref, bandpass_array, n=10):
        """Returns the frequency scaling factor given a frequency array and a 
        bandpass profile.

        Makes use of the mixing matrix implementation from Commander3. For 
        more information, see section 4.2 in https://arxiv.org/abs/2011.05609.

        TODO: FIX 2D interp case.
        TODO: Test that this actually works for real components. Use the fact that 
        dicts are ordered in >3.6

        Args:
        -----
        nu_array : astropy.units.quantity.Quantity
            Frequencies corresponding to the bandpass weights.
        nu_ref : tuple, list, np.ndarray
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
            freq_scaling = self.get_freq_scaling(nu_array, nu_ref, **scalars)
            integral = np.trapz(freq_scaling*bandpass_array, nu_array)
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
                    freq_scaling = self.get_freq_scaling(nu_array, nu_ref, 
                                                         **spectrals)
                    integrals.append(
                        np.trapz(freq_scaling*bandpass_array, nu_array)
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
            integral_unit = (bandpass_array.unit * nu_array.unit)
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
                    freq_scaling = self.get_freq_scaling(nu_array, nu_ref, 
                                                         **spectrals)
                    integral = np.trapz(freq_scaling*bandpass_array, nu_array)
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
        main_repr += '(amp, nu_ref, '
        for spectral in self._spectrals:
            main_repr += f'{spectral}, '
        main_repr = main_repr[:-2]
        main_repr += ')'

        return main_repr




class PowerLaw(Component):
    """PowerLaw component class.

    Args:
    -----
    comp_name : str
        Name/label of the component that will uses this model, e.g 'dust'. 
        When added to a cosmoglobe.Model, the attribute name will be the 
        comp_name.
    amp : astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        PowerLaw IQU amplitude map.
    nu_ref : tuple, list, np.ndarray
        Reference frequencies for the amplitude map. Each array 
        element must be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    beta: astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        PowerLaw IQU beta map.

    """
    def __init__(self, comp_name, amp, nu_ref, beta):
        super().__init__(comp_name, amp, nu_ref, beta=beta)


    def get_freq_scaling(self, nu, nu_ref, beta):
        """Computes the frequency scaling from the reference frequency nu_ref 
        to an arbitrary frequency nu, which depends on the spectral parameter
        beta.

        Args:
        -----
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. Must be in si values.      
        nu_ref : tuple, list, np.ndarray
            Reference frequencies for the amplitude map.
        beta : numpy.ndarray
            Synch beta map. Must be dimensionless.
            
        Returns:
        --------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        scaling = (nu/nu_ref)**beta
        return scaling


    
class ModifiedBlackBody(Component):
    """Modified BlackBody (MBB) component class.

    Args:
    -----
    comp_name : str
        Name/label of the component that will uses this model, e.g 'dust'. 
        When added to a cosmoglobe.Model, the attribute name will be the 
        comp_name.
    amp : astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        MBB IQU amplitude map.
    nu_ref : tuple, list, np.ndarray
        Reference frequencies for the synch amplitude map. Each array 
        element must be an astropy.Quantity, with unit Hertz, e.g u.GHz.
    beta: astropy.units.quantity.Quantity, cosmoglobe.IQUMap
        MBB IQU beta map.

    """
    def __init__(self, comp_name, amp, nu_ref, beta, T):
        super().__init__(comp_name, amp, nu_ref, beta=beta, T=T)


    def get_freq_scaling(self, nu, nu_ref, beta, T):
        """Computes the frequency scaling from the reference frequency nu_ref 
        to an arbitrary frequency nu, which depends on the spectral parameters
        beta and T.

        Args:
        -----
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : tuple, list, np.ndarray
            Reference frequencies for the amplitude map.
        beta : numpy.ndarray
            MBB beta map. Must be dimensionless.
        T : numpy.ndarray
            MBB temperature map with unit K.

        Returns:
        --------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        scaling = blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)
        scaling *= (nu/nu_ref)**(beta-2)

        return scaling