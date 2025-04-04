import numpy as np
from cosmoglobe.tod_tools import CommanderDataAdapter
from dataclasses import dataclass
from astropy.time import Time
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic, get_body, AltAz
from astropy import units as u
from astropy.io import fits
import healpy as hp
import glob


@dataclass
class CommCHIPASSDataAdapter(CommanderDataAdapter):

    def __post_init__(self):
        self.currchunk = None
        self.bands = ['1395']
        self.nside = 1024
        self.dets = {
            '1395': [f'{el:02}' for el in np.arange(1, 14)]
        }
        self.fsamp = {'1395':8}
        self.npsi = 1
        self.num_chunks = 50690
        self.num_segments = self.num_chunks // 100 + 1

    def get_num_segments(self, band):
        return self.num_segments

    def get_experiment_name(self):
        return 'chipass'

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
            iterator = range((segment-1)*100+1, segment*100+1)
        else:
            iterator = range((segment-1)*100+1, self.num_chunks+1)
        for idx in iterator:
            to_be_added = True
            '''
            for det in self.dets[band]:
                if not os.path.exists(f):
                    to_be_added = False
                    break
            '''
            if to_be_added:
                out_indices.append(idx)
        return out_indices

    def get_bands(self):
        return self.bands

    def set_chunk_index(self, chunk_idx: int):
        self.currchunk = chunk_idx
        self.chunk_starttime = None

    def get_chunk_data(self, band, detector):
        data_dir = '/mn/stornext/d16/cmbco/ola/chipass/l1_data/'
        file_list = sorted(glob.glob(data_dir + "/**/*.sdfits", recursive=True))
        f = file_list[self.currchunk-1]
        data_raw = fits.open(f)
        # time
        date = np.array(data_raw[1].data['date-obs']).reshape(-1,13)[:, 0]
        t0 = Time(date, format='isot', scale='utc').mjd
        tt = data_raw[1].data['time'].reshape(-1,13)[:, 0]  # time, in seconds, since start of day (UTC)
        nt = len(tt)  # number of samples in time
        t = t0 + tt / 3600.0 / 24
        # Pointing info 
        az = data_raw[1].data['AZIMUTH'].reshape(-1,13)
        el = data_raw[1].data['ELEVATIO'].reshape(-1,13)
        # Lat/Lon/h (WGS84) of Parkes telescope 32.9984064, 148.2635101, 414.80
        # (https://www.narrabri.atnf.csiro.au/observing/users_guide/html/chunked/apg.html)
        loc = coord.EarthLocation(lon=148.2635101 * u.deg, lat=-32.9984064 * u.deg, height=414.80 * u.m)
        time = Time(t, format='mjd')
        AltAz = SkyCoord(alt=el*u.deg, az=az*u.deg, obstime=time[:, None], frame='altaz', location=loc)
        point = np.array([AltAz.galactic.l.degree, AltAz.galactic.b.degree, np.zeros_like(AltAz.galactic.b.degree)]).transpose((1, 2, 0))
        point = point.astype(np.float32)
        lon = np.copy(point[:, int(detector)-1, 0])
        lat = np.copy(point[:, int(detector)-1, 1])
        pix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
        # tod
        tod = data_raw[1].data['Data'][:,0,0,:,:]
        tod = np.reshape(tod, (-1, 13, 2, 1024))
        tsys = data_raw[1].data['Tsys']
        tsys = np.reshape(tsys, (-1, 13, 2))
        flag = data_raw[1].data['flagged'][:,0,0,:,:]
        flag = np.reshape(flag, (-1, 13, 2, 1024))
        mask = np.array(1.0 - flag).astype(float)
        ratio = tod[:, :, :, :] / tsys[:, :, :, None]
        n_med = 25
        n_half = n_med // 2
        n = ratio.shape[0]
        bst = np.zeros_like(ratio)
        for i in range(n):
            low = max(i-n_half, 0)
            hi = min(i+n_half, n)
            bst[i] = np.nanmedian(ratio[low:hi], axis=0)
        n = tsys.shape[0]
        bt = np.zeros_like(tsys)
        for i in range(n):
            low = max(i-n_half, 0)
            hi = min(i+n_half, n)
            bt[i] = np.nanmedian(tsys[low:hi], axis=0)
        s_prime = tod / bst - bt[:, :, :, None]  # Calibrated data in Jy
        # bandpass integration (using median for now)
        s_cont = np.zeros((nt, 13, 2, 4))
        mask[:, :, :, 85:125] = 0  # remove 
        mask[(mask == 0)] = np.nan
        for i in range(4):
            s_cont[:, :, :, i] = np.nanmedian(mask[:, :, :, i*256:(i+1)*256] * s_prime[:, :, :, i*256:(i+1)*256], axis=3)
        # average over four bandpasses
        tod_new = np.sqrt(s_cont[:, :, 0] ** 2 + s_cont[:, :, 1] ** 2).mean(2)
        # get data for this detector
        tod = np.copy(tod_new[:, int(detector)-1])

        if self.chunk_starttime is None:
            self.chunk_starttime = Time(date[0], format='isot', scale='utc') + tt[0]*u.s
            self.chunk_endtime = Time(date[0], format='isot', scale='utc') + tt[-1]*u.s

        flag_tot = np.zeros(len(t))
        flag_tot[~np.isfinite(lon)] += 2**0
        flag_tot[~np.isfinite(tod)] += 2**1

        tod[flag_tot != 0] = 0
        lon[flag_tot != 0] = 0
        lat[flag_tot != 0] = 0

        psi = np.zeros_like(tod)
        
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
