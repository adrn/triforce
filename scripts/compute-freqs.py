# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import superfreq

def dat_iterator(filename):
    """
    Iterate through chunks of the output file.
    """

    with open(filename) as f:
        rows = []
        start_caching = False
        for i,line in enumerate(f):
            if i <= 5:
                continue

            row = map(float, line.split())

            if row[0] == 0. and start_caching:
                yield np.array(rows)
                start_caching = False
                rows = []

            if row[0] == 0:
                start_caching = True

            rows.append(row)

def read_ntimesteps_nparticles(filename):
    """
    Read number of timesteps and particles from header of file
    """

    # read number of timesteps, particles from header
    with open(filename) as f:
        for i,line in enumerate(f):
            if i == 3:
                nparticles = int(line.strip())**2

            elif i == 5:
                ntimesteps = int(line.strip())

            if i >= 5:
                break

    return ntimesteps, nparticles

def cartesian_to_poincare_polar(w):
    r"""
    Convert an array of 4D Cartesian positions to Poincaré
    symplectic polar coordinates. These are similar to polar
    coordinates.

    Parameters
    ----------
    w : array_like
        Input array of 4D Cartesian phase-space positions. Should have
        shape ``(norbits,4)``.

    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        Points represented in 4D Poincaré polar coordinates.

    """

    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+2] + w[...,1]*w[...,1+2]) / R
    vPhi = w[...,0]*w[...,1+2] - w[...,1]*w[...,0+2]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    new_w = np.vstack((R.T, pp_phi.T,
                       vR.T, pp_phidot.T)).T
    return new_w

def main(filename, overwrite=False, plot=False):
    if not os.path.exists(filename):
        raise IOError("File '{}' does not exist -- are you sure you specified"
                      " the correct path?")

    plotpath = os.path.join("plots", os.path.splitext(os.path.basename(filename))[0])
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)

    basename = os.path.splitext(filename)[0]
    save_file = "{}_freqs.npy".format(basename)
    if os.path.exists(save_file) and overwrite:
        os.remove(save_file)

    # number of degrees of freedom of the potential (restricted to plane)
    ndof = 2

    # number of particles, number of timesteps
    ntimesteps,nparticles = read_ntimesteps_nparticles(filename)
    ntimesteps += 2 # for some reason?

    # container for all frequencies
    all_freqs = np.zeros((nparticles,ndof))

    if plot:
        fig_xy,axes_xy = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        axes_xy[0].set_xlim(-2,2)
        axes_xy[0].set_ylim(-2,2)
        axes_xy[0].set_xlabel("$x$")
        axes_xy[0].set_ylabel("$y$")
        axes_xy[1].set_xlabel("$x$")
        axes_xy[0].set_title("rotating frame", fontsize=18)
        axes_xy[1].set_title("intertial frame", fontsize=18)
        fig_xy.tight_layout()

    for i,block in enumerate(dat_iterator(filename)):
        logger.debug("- Reading block for particle {}".format(i))

        if len(block) < ntimesteps:
            logger.debug("-- Skipping particle because number of steps < ntimesteps.")

        # grab time array and x,y,vx,vy
        t = block[:,0]
        w = block[:,1:5]

        # non-rotating frame
        xy = np.zeros_like(w[:,:2])
        xy[:,0] = w[:,0]*np.cos(t) - w[:,1]*np.sin(t)
        xy[:,1] = w[:,0]*np.sin(t) + w[:,1]*np.cos(t)

        vxy = np.zeros_like(w[:,2:4])
        vxy[:,0] = -w[:,2]*np.sin(t) - w[:,3]*np.cos(t)
        vxy[:,1] = w[:,2]*np.cos(t) - w[:,3]*np.sin(t)

        if plot:
            pts1, = axes_xy[0].plot(w[:,0], w[:,1], color='k')
            pts2, = axes_xy[1].plot(xy[:,0], xy[:,1], color='k')
            fig_xy.savefig(os.path.join(plotpath, "{}.png".format(i)), dpi=300)
            pts1.remove()
            pts2.remove()

        w = np.hstack((xy,vxy))
        w = cartesian_to_poincare_polar(w)
        fs = [(w[:,j] + 1j*w[:,j+ndof]) for j in range(ndof)]

        # now compute the frequencies
        sf = superfreq.SuperFreq(t=t, p=4)
        try:
            freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, min_freq=1E-6,
                                                             min_freq_diff=1E-6, nintvec=15)
        except ValueError:
            freqs = np.ones(ndof)*np.nan

        # # verify that reconstructed time series looks like input
        # for k in range(ndof):
        #     sub_tbl = tbl[tbl['idx'] == k]
        #     xx = np.sum(sub_tbl['A'][None] * np.exp(1j * sub_tbl['freq'][None] * t[:,None]), axis=-1)

        #     pl.clf()
        #     pl.plot(t, xx.real)
        #     pl.plot(t, fs[k].real)
        #     pl.show()
        # sys.exit(0)

        logger.debug("-- Computed freqeuncies: {}".format(freqs))
        all_freqs[i] = freqs

    if not os.path.exists(save_file):
        np.save(save_file, all_freqs)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="Overwrite any output.")

    parser.add_argument("-f", "--file", dest="filename", required=True,
                        type=str, help="Name of the file to compute frequencies for.")
    parser.add_argument("-p", dest="plot", action="store_true", default=False,
                        help="Plot or not")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.filename, overwrite=args.overwrite, plot=args.plot)
