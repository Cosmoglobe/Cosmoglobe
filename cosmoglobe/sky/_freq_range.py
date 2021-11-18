from dataclasses import dataclass

from astropy.units import Quantity


@dataclass
class FrequencyRange:
    """
    Class representing a frequency range, outside of which a components emission
    can be assumed to be negligable.
    """
    
    lower: Quantity
    upper: Quantity

    def __contains__(self, freqs: Quantity) -> bool:
        """Returns True, if `freqs` is within the boundaries and False if else."""

        if freqs.ndim > 0:
            return any(self.lower < freq < self.upper for freq in freqs)

        return self.lower < freqs < self.upper
