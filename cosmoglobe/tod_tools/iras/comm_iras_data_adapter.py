import numpy as np
from cosmoglobe.tod_tools import CommanderDataAdapter
from dataclasses import dataclass
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK4
from astropy.coordinates import HeliocentricMeanEcliptic, get_body, solar_system_ephemeris
from astropy import units as u
import healpy as hp
import os

GC = SkyCoord(l=0*u.degree, b=0*u.degree, frame='galactic')


@dataclass
class CommIRASDataAdapter(CommanderDataAdapter):

    def __post_init__(self):
        self.iras_data_path = Path("/mn/stornext/d5/data/duncanwa/IRAS/sopobs_data")
        self.currchunk = None
        self.bands = ['12', '25', '60', '100']
        self.nside = 2048
        self.t0 = Time('1981-01-01', scale='utc')
        self.dets = {
            '12': [f'{el:02}' for el in np.concatenate((np.arange(23, 31), np.arange(48, 54)))],
            '25': [f'{el:02}' for el in np.concatenate((np.array([16,18,19,21, 22]), np.arange(40,46)))],
            '60': [f'{el:02}' for el in np.concatenate((np.array([8,9,10,13,14,15]), np.array([32,33,34,35,37])))],
            '100': [f'{el:02}' for el in np.concatenate((np.arange(1, 8), np.arange(56, 62)))]
        }
        self.fsamp ={'12':16, '25':16, '60':8, '100':4}
        self.npsi = 8  # For IRAS this shouldn't matter
        self.Omegas = {1: 14.5,
                       2: 12.7,
                       3: 13.0,
                       4: 11.53,
                       5: 12.0,
                       6: 12.4,
                       7:12.6,
                       8:7.2,
                       9:6.7,
                       10:6.6,
                       11:2.8,
                       12:4.3,
                       13:6.6,
                       14:6.1,
                       15:6.2,
                       16:3.5,
                       18:3.6,
                       19:2.8,
                       21:2.8,
                       22:3.1,
                       23:2.9,
                       24:3.0,
                       25:3.2,
                       26:1.2,
                       27:2.0,
                       28:3.1,
                       29:2.5,
                       30:2.8,
                       55:7.1,
                       56:14.0,
                       57:13.2,
                       58:11.2,
                       59:11.7,
                       60:13.3,
                       61:13.5,
                       62:10.6,
                       31:2.1,
                       32:6.4,
                       33:5.9,
                       34:6.5,
                       35:6.3,
                       37:6.6,
                       38:3.9,
                       39:1.4,
                       40:3.1,
                       41:3.1,
                       42:3.4,
                       43:3.2,
                       44:3.2,
                       45:3.2,
                       46:2.4,
                       47:0.77,
                       48:3.1,
                       49:2.9,
                       50:3.0,
                       51:2.7,
                       52:2.5,
                       53:2.8,
                       54:2.0}

        self.num_chunks = 5787
        self.num_segments = self.num_chunks // 10


    def get_num_segments(self, band):
#        return 3
        return self.num_segments

    def get_experiment_name(self):
        return 'iras'

    def get_npsi(self):
        return self.npsi

    def get_nside(self, band):
        return self.nside

    def get_polangs(self, band):
        return [0 for det in self.dets[band]]

    def get_mbangs(self, band):
        return [0 for det in self.dets[band]]

    def get_fsamp(self, band):
        return self.fsamp[band]

    def get_detector_names(self, band):
        return self.dets[band]

    def get_chunk_indices(self, band, segment):
        out_indices = []
        if segment != self.num_segments:
            iterator = range((segment-1)*10+1, segment*10+1)
        else:
            iterator = range(segment*10+1, self.num_chunks+1)
        for idx in iterator:
            to_be_added = True
            for det in self.dets[band]:
                if idx == 273:
                    to_be_added = False
                    break
                f = f'{self.iras_data_path}/det_{det}/sopobs_{idx:04}.npy'
                if not os.path.exists(f):
                    to_be_added = False
                    break
            if to_be_added:
                out_indices.append(idx)
        return out_indices

    def get_bands(self):
#        return self.bands[:2]
        return self.bands

    def set_chunk_index(self, chunk_idx: int):
        self.currchunk = chunk_idx
        self.chunk_starttime = None

    def get_chunk_data(self, band, detector):

        f = f'{self.iras_data_path}/det_{detector}/sopobs_{self.currchunk:04}.npy'
        if os.path.exists(f):
            t, lon, lat, tod = np.load(f)
        else:
            # Should not really happen now, since we are doing this filtering earlier
            return [None] * 4

        if self.chunk_starttime is None:
            self.chunk_starttime = t[0]*u.s + self.t0
            self.chunk_endtime = t[-1]*u.s + self.t0

        tod /= self.Omegas[int(detector)] * 1e-7

        flag_tot = np.zeros(len(t))
        flag_tot[~np.isfinite(lon)] += 2**0
        flag_tot[~np.isfinite(tod)] += 2**1

        tod[flag_tot != 0] = 0
        lon[flag_tot != 0] = 0
        lat[flag_tot != 0] = 0

        psi = np.zeros_like(tod)
        
        good_data = flag_tot == 0
        sc = SkyCoord(ra=lon[good_data], dec=lat[good_data], unit='deg',
                equinox='B1950.0', obstime='J1983.5', frame=FK4)
        coords = sc.transform_to(GC)
        lon[good_data] = coords.l.value
        lat[good_data] = coords.b.value
        pix = hp.ang2pix(self.nside, lon, lat, lonlat=True) 

        return tod, pix, psi, flag_tot


    def get_chunk_start_time(self):
        return self.chunk_starttime

    def get_chunk_end_time(self):
        return self.chunk_endtime

    def get_version(self):
        return 0

    def get_chunk_start_satpos(self):
        earth_pos = get_body("earth", Time(self.chunk_starttime.mjd,
                                           format="mjd")).transform_to(
                                               HeliocentricMeanEcliptic)
        earth_pos = earth_pos.cartesian.xyz.to(u.AU).transpose()
        return earth_pos

    def get_chunk_end_satpos(self):
        earth_pos = get_body("earth", Time(self.chunk_endtime.mjd,
                                           format="mjd")).transform_to(
                                               HeliocentricMeanEcliptic)
        earth_pos = earth_pos.cartesian.xyz.to(u.AU).transpose()
        return earth_pos

    def get_chunk_start_earthpos(self):
        return self.get_chunk_start_satpos()

    def get_chunk_end_earthpos(self):
        return self.get_chunk_end_satpos()

    def get_chunk_satvel(self):
        return [0, 0, 0]

    def get_should_compress_tods(self):
        return False

    def get_gain(self, band, detector):
        return 1

    def get_alpha(self, band, detector):
        return -2

    def get_fknee(self, band, detector):
        return 0.1

    def get_sigma0(self, band, detector):
        return 1
