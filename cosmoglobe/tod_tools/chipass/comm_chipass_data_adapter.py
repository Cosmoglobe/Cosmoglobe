import numpy as np
from cosmoglobe.tod_tools import CommanderDataAdapter
from dataclasses import dataclass
from astropy.time import Time
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, FK5, HeliocentricMeanEcliptic, get_body
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
        t = t0 + tt / 3600.0 / 24
        nt = len(tt)  # number of integrations in scan
        # Pointing info 
        ra = data_raw[1].data['CRVAL3'].reshape(nt, 13)
        dec = data_raw[1].data['CRVAL4'].reshape(nt, 13)
        sc = SkyCoord(ra=ra[:, int(detector)-1], dec=dec[:, int(detector)-1], unit='deg', frame=FK5, equinox='J2000.0')
        gc = sc.transform_to('galactic')
        lon = gc.l.degree
        lat = gc.b.degree
        pix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
        # tod
        tod = data_raw[1].data['Data'][:,0,0,:,:]
        tod = np.reshape(tod, (nt, 13, 2, 1024))
        flag = data_raw[1].data['flagged'][:,0,0,:,:]
        flag = np.reshape(flag, (nt, 13, 2, 1024))
        mask = np.array(1.0 - flag).astype(float)
        # calibrate new
        '''
        tsys = data_raw[1].data['Tsys']
        tsys = np.reshape(tsys, (nt, 13, 2))
        tod[np.where(mask == 0.0)] = np.nan # added: apply mask before calculating running median filter
        ratio = tod[:, :, :, :] / tsys[:, :, :, None]
        n_med = 10
        n_half = n_med // 2
        n = ratio.shape[0]
        run_med = np.zeros_like(ratio)
        for i in range(n):
            low = max(i-n_half, 0)
            hi = min(i+n_half+1, n)
            run_med[i] = np.nanmedian(ratio[low:hi], axis=0)
        bst = np.nanmin(run_med, axis=0)
        n = tsys.shape[0]
        run_med = np.zeros_like(tsys)
        for i in range(n):
            low = max(i-n_half, 0)
            hi = min(i+n_half+1, n)
            run_med[i] = np.nanmedian(tsys[low:hi], axis=0)
        bt = np.nanmin(run_med, axis=0)
        s_prime = tod / bst[None, :, :, :] - bt[None, :, :, None]  # Calibrated data in Jy
        '''
        s_prime = np.copy(tod) # do not calibrate, use raw TOD
        # Remove 21 cm channels
        mask[:, :, :, 85:125] = 0  # remove 
        # remove additional bands
        mask[:, :, :, 0:10] = 0 # 2:10
        mask[:, :, :, 151:159] = 0
        mask[:, :, :, 236:244] = 0
        mask[:, :, :, 294:298] = 0
        mask[:, :, :, 390:398] = 0
        mask[:, :, :, 423:427] = 0
        mask[:, :, :, 483:491] = 0
        mask[:, :, :, 727:735] = 0
        mask[:, :, :, 972:980] = 0
        mask[(mask == 0)] = np.nan
        # bandpass integration (using median for now)
        # divide channels into four bands and get the median in each
        s_cont = np.zeros((nt, 13, 2, 4))
        for i in range(4):
            s_cont[:, :, :, i] = np.nanmedian(mask[:, :, :, i*256:(i+1)*256] * s_prime[:, :, :, i*256:(i+1)*256], axis=3)
        # combine the polarizations and average over the four bandpasses
        tod_new = np.sqrt(s_cont[:, :, 0] ** 2 + s_cont[:, :, 1] ** 2).mean(2)
        # get data for this detector
        tod = np.copy(tod_new[:, int(detector)-1])
        # convert units from Jy beam^-1 to MJy sr^-1 based on beam size (FWHP approx 14.3 arcmin for all beams)
        #tod *= 0.057793
        # convert units from Jy beam^-1 to mK and apply offset following Calabretta2014 -- 0.44 K/(Jy beam^-1), + 3.3 K offset
        #tod *= 440.0 # mK
        #tod += 3300.0

        if self.chunk_starttime is None:
            self.chunk_starttime = Time(date[0], format='isot', scale='utc') + tt[0]*u.s
            self.chunk_endtime = Time(date[0], format='isot', scale='utc') + tt[-1]*u.s

        flag_tot = np.zeros(len(t))
        flag_tot[~np.isfinite(lon)] += 2**0
        flag_tot[~np.isfinite(tod)] += 2**1
        #if np.all(flag_tot): print('fully-masked scan', f)

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
