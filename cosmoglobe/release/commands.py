import time
import os
import numpy as np
import sys
import click
from cosmoglobe.release.tools import *

@click.group()
def commands():
    pass

@commands.command()
@click.argument("filename", type=click.STRING)
@click.argument("min", type=click.INT)
@click.argument("max", type=click.INT)
@click.argument("binfile", type=click.STRING)
def dlbin2dat(filename, min, max, binfile):
    """
    Outputs binned powerspectra averaged over a range of output samples.
    Filename Dl_[signal]_binned.dat.
    """
    signal = "cmb/Dl"

    import h5py

    dats = []
    with h5py.File(filename, "r") as f:
        for sample in range(min, max + 1):
            # Get sample number with leading zeros
            s = str(sample).zfill(6)

            # Get data from hdf
            data = f[s + "/" + signal][()]
            # Append sample to list
            dats.append(data)
    dats = np.array(dats)

    binned_data = {}
    possible_signals = ["TT","EE","BB","TE","EB","TB",]
    with open(binfile) as f:
        next(f)  # Skip first line
        for line in f.readlines():
            line = line.split()
            signal = line[0]
            if signal not in binned_data:
                binned_data[signal] = []
            signal_id = possible_signals.index(signal)
            lmin = int(line[1])
            lmax = int(line[2])
            ellcenter = lmin + (lmax - lmin) / 2
            # Saves (ellcenter, lmin, lmax, Dl_mean, Dl_stddev) over samples chosen
            binned_data[signal].append([ellcenter, lmin, lmax, np.mean(dats[:, signal_id, lmin], axis=0,), np.std(dats[:, signal_id, lmin], axis=0,),])

    header = f"{'l':22} {'lmin':24} {'lmax':24} {'Dl':24} {'stddev':24}"
    for signal in binned_data.keys():
        np.savetxt("Dl_" + signal + "_binned.dat", binned_data[signal], header=header,)


@commands.command()
@click.argument("label", type=click.STRING)
@click.argument("freqs", type=click.FLOAT, nargs=-1)
@click.argument("nside", type=click.INT,)
@click.option("-cmb", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-synch", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-dust", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-ff", type=click.Path(exists=True), help="Include resampled chain file",)
@click.option("-ame", type=click.Path(exists=True), help="Include resampled chain file",)
#@click.option("-skipcopy", is_flag=True, help="Don't copy full .h5 file",)
def generate_sky(label, freqs, nside, cmb, synch, dust, ff, ame):
    """
    Generate sky maps from separate input maps.
    Reference frequencies from BP: CMB 1, SYNCH 30, DUST 545 353, FF 40, AME 22,
    Example:
    "c3pp generate-sky test 22 30 44 1024 -cmb BP_cmb_IQU_full_n1024_v1.0.fits -synch BP_synch_IQU_full_n1024_v1.0.fits"
    # Todo smoothing and nside?
    """
    import healpy as hp
    import numpy as np
    # Generate sky maps
    A = 1 # Only relative scaling
    for nu in freqs:
        filename = f"{label}_{nu}_n{nside}_generated.fits"
        print(f"Generating {filename}")
        data = np.zeros((3, hp.nside2npix(nside)))
        for pl in range(3):

            if cmb:
                pl_data  = hp.read_map(cmb, field=pl, verbose=False,)

            if synch:
                scaling  = fgs.lf(nu, A, betalf=-3.11, nuref=30.)
                print(hp.read_map(synch, field=pl, verbose=False)*scaling)
                pl_data += hp.read_map(synch, field=pl, verbose=False)*scaling

            if dust:
                if pl > 0:
                    scaling = fgs.dust(nu, A, beta=1.6, Td=18.5, nuref=353.,)
                else:
                    scaling = fgs.dust(nu, A, beta=1.6, Td=18.5, nuref=545.,)
                pl_data += hp.read_map(dust, field=pl, verbose=False,)*scaling

            if pl == 0:
                if ff:
                    scaling  = fgs.ff(nu, A, Te=7000., nuref=40.) 
                    pl_data += hp.read_map(ff, field=pl, verbose=False,)*scaling

                if ame:
                    scaling  = fgs.sdust(nu, A, nu_p=21, nuref=22.)
                    pl_data += hp.read_map(ame, field=pl, verbose=False,)*scaling

            data[pl] = pl_data
        hp.write_map(filename, data, dtype=None)
