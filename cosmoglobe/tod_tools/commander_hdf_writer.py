# Defines a class, CommanderHDFWriter, which takes a CommanderDataAdapter
# object and creates an HDF file based on it. The CommanderDataAdapter object
# is experiment-specific but should always include the functions that is
# defined in the base class, which is what this class expects.
from dataclasses import dataclass
from cosmoglobe.tod_tools import CommanderDataAdapter
from cosmoglobe.tod_tools import TODLoader
#from tod_tools import commander_tod as comm_tod
import numpy as np
import multiprocessing as mp


@dataclass(eq=False)
class CommanderHDFWriter:
    comm_adapter: CommanderDataAdapter

    def __post_init__(self):
        self.should_compress_tods = self.comm_adapter.get_should_compress_tods()
        self.npsi = self.comm_adapter.get_npsi()
        self.huffman = ['huffman', {'dictNum':1}]
        self.psi_digitize = ['digitize', {'min':0, 'max':2*np.pi,'nbins':self.npsi, 'offset':1}] # Will give values from 1 to npsi, inclusive.
        self.hufftod = ['huffman', {'dictNum':2}]
        self.bands = self.comm_adapter.get_bands()
        self.gain = {}
        self.fknee = {}
        self.alpha = {}
        self.sigma0 = {}
        self.nside = {}
        self.polangs = {}
        self.mbangs = {}
        self.fsamp = {}


    def write_hdf_files(self, hdf_output_dir, bands=None, overwrite=False,
                        num_processes=None):

        if bands is None:
            bands = self.comm_adapter.get_bands()

        if num_processes is None:
            filelists = dict([(band, {}) for band in bands])
            pool = None
        else:
            pool = mp.Pool(processes=num_processes)
            manager = mp.Manager()
            filelists = dict([(band, manager.dict()) for band in bands])

        ctod = TODLoader(hdf_output_dir,
                         self.comm_adapter.get_experiment_name(),
                         version=self.comm_adapter.get_version(),
                         dicts=filelists,
                         overwrite=overwrite)

        for band in bands:
            self.nside[band] = self.comm_adapter.get_nside(band)
            self.polangs[band] = self.comm_adapter.get_polangs(band)
            self.mbangs[band] = self.comm_adapter.get_mbangs(band)
            self.fsamp[band] = self.comm_adapter.get_fsamp(band)
            self._process_band(band, ctod, pool)

        if pool is not None:
            pool.close()
            pool.join()

        ctod.make_filelists()


    def _process_band(self, band: str, ctod: TODLoader, pool=None):
        detectors = self.comm_adapter.get_detector_names(band)
        self.gain[band] = {}
        self.fknee[band] = {}
        self.alpha[band] = {}
        self.sigma0[band] = {}

        for detector in detectors:
            self.gain[band][detector] = self.comm_adapter.get_gain(band, detector)
            self.alpha[band][detector] = self.comm_adapter.get_alpha(band, detector)
            self.fknee[band][detector] = self.comm_adapter.get_fknee(band, detector)
            self.sigma0[band][detector] = self.comm_adapter.get_sigma0(band, detector)

        nsegments = self.comm_adapter.get_num_segments(band)
        from tqdm import tqdm
        if pool is None:
            for segment in range(1, nsegments+1):
                self._process_hdf_segment(segment, band, detectors, ctod)
        else:
            [pool.apply_async(self._process_hdf_segment, (segment, band,
                                                          detectors, ctod)) for
             segment in range(1, nsegments+1)]

    def _process_hdf_segment(self, segment: int, band: str, detectors:
                             list[str], ctod: TODLoader):
        ctod.init_file(band, segment, mode='w') # assuming that segment is the segment number
        chunks = self.comm_adapter.get_chunk_indices(band, segment)
        for chunk in chunks:
            self.comm_adapter.set_chunk_index(chunk)
            self._process_chunk(chunk, band, detectors, ctod)

        prefix = 'common'
        ctod.add_field(prefix + '/det', ','.join(list(detectors)))
        ctod.add_field(prefix + '/nside', self.nside[band])
        ctod.add_field(prefix + '/polang', self.polangs[band])
        ctod.add_field(prefix + '/mbang', self.mbangs[band])
        ctod.add_field(prefix + '/fsamp', self.fsamp[band])
        ctod.finalize_file()
        ctod.outFile.close()

    def _process_chunk(self, chunk: int, band: str, detectors: list[str], ctod:
                       TODLoader):
        self.chunk_size = None # This can't always be queried before loading a specific chunk, so we set it later during chunk processing.

        for detector in detectors:
            ctod = self._process_detector(detector, band, chunk, ctod)
        ctod.add_field(f"{chunk:06d}/common/ntod", self.chunk_size) # Should have been updated during the first 'process_detector' call.

        chunk_start_time = self.comm_adapter.get_chunk_start_time() # Format='isot'
        ctod.add_field(f"{chunk:06d}/common/time", [chunk_start_time.mjd, 0, 0])

        chunk_end_time = self.comm_adapter.get_chunk_end_time() # Format='isot'
        ctod.add_field(f"{chunk:06d}/common/time_end", [chunk_end_time.mjd, 0, 0])

        chunk_start_satpos = self.comm_adapter.get_chunk_start_satpos() # In galactic coordinates, cartesian, in m
        ctod.add_field(f"{chunk:06d}/common/satpos", chunk_start_satpos)

        chunk_end_satpos = self.comm_adapter.get_chunk_end_satpos() # In galactic coordinates, cartesian, in m
        ctod.add_field(f"{chunk:06d}/common/satpos_end", chunk_end_satpos)

        chunk_start_earthpos = self.comm_adapter.get_chunk_start_earthpos()
        ctod.add_field(f"{chunk:06d}/common/earthpos", chunk_start_earthpos)

        chunk_end_earthpos = self.comm_adapter.get_chunk_end_earthpos()
        ctod.add_field(f"{chunk:06d}/common/earthpos_end", chunk_end_earthpos)

        chunk_vel = self.comm_adapter.get_chunk_satvel() # m/s
        ctod.add_field(f"{chunk:06d}/common/vsun", chunk_vel)

        ctod.finalize_chunk(f"{chunk:06d}")

    def _process_detector(self, detector: str, band: str, chunk: int, ctod:
                          TODLoader):
        prefix = f"{chunk:06d}/{detector}"
        tod_arr, pix_arr, psi_arr, flag_arr = self.comm_adapter.get_chunk_data(band, detector)

        if tod_arr is None:
            raise NotImplementedError()

        if self.chunk_size is None:
            self.chunk_size = len(tod_arr)

        if not self.should_compress_tods:
            ctod.add_field(prefix + '/tod', tod_arr)
        else:
            raise NotImplementedError()

        ctod.add_field(prefix + '/pix', pix_arr, [self.huffman])

        ctod.add_field(prefix + '/psi', psi_arr, [self.psi_digitize, self.huffman])

        ctod.add_field(prefix + '/flag', flag_arr, [self.huffman])

        scalars = [
            self.gain[band][detector],
            self.sigma0[band][detector],
            self.fknee[band][detector],
            self.alpha[band][detector]
        ]

        ctod.add_field(prefix + '/scalars', scalars)
        outp = np.array([0, 0]) # Right now I don't know what to do about this one..
        ctod.add_field(prefix + '/outP', outp)

        return ctod
