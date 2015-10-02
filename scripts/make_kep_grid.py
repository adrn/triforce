# coding: utf-8

""" Generate a grid of initial conditions at constant Jacobi energy """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np

# Project
from triforce import project_path

def main(ngrid, q, ecc=0., nu=0., max_xy=3., output_path=None,
         run_name=None, overwrite=False, plot=False):
    """
    """

    # create path
    if output_path is None:
        output_path = os.path.join(project_path, "output")

        if run_name is None:
            run_name = "n{}_kepdisk_q{:.0e}_e{:.0e}_nu{:.0e}".format(ngrid, q, ecc, nu)
        path = os.path.join(output_path, run_name)
    else:
        path = output_path

    logger.info("Caching to: {}".format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    # path to initial conditions cache
    w0path = os.path.join(path, 'w0.npy')

    if os.path.exists(w0path) and overwrite:
        os.remove(w0path)

    if not os.path.exists(w0path):
        # generate initial conditions

        # disc
        r = np.sqrt(np.random.uniform(0.,max_xy**2,ngrid*ngrid))
        phi = np.random.uniform(0.,2*np.pi,ngrid*ngrid)
        xyz = np.zeros((len(r),3))
        xyz[:,0] = r*np.cos(phi)
        xyz[:,1] = r*np.sin(phi)

        u2 = q/(1.+q)
        u1 = 1.0-u2
        r1 = np.sqrt((xyz[:,0]+u2)**2 + xyz[:,1]**2 + xyz[:,2]**2)
        r2 = np.sqrt((xyz[:,0]-u1)**2 + xyz[:,1]**2 + xyz[:,2]**2)

        r = np.sqrt(np.sum(xyz**2, axis=-1))
        vmag = np.sqrt(u1/r1 + u2/r2) - r
        vx = -vmag * xyz[:,1]/r
        vy = vmag * xyz[:,0]/r

        vxyz = np.zeros_like(xyz)
        vxyz[:,0] = vx
        vxyz[:,1] = vy

        w0 = np.hstack((xyz,vxyz))

    else:
        w0 = np.load(w0path)
        logger.info("Initial conditions file already exists!\n\t{}".format(w0path))

    if plot:
        fig,ax = pl.subplots(1,1,figsize=(8,8))
        ax.plot(w0[:,0], w0[:,1], ls='none', marker=',', color='k')
        fig.savefig(os.path.join(path, 'w0.png'), dpi=300)

    logger.info("Number of initial conditions: {}".format(len(w0)))
    np.save(w0path, w0)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")

    parser.add_argument("-p","--output-path", dest="output_path", default=None,
                        help="Path to the 'output' directory.")
    parser.add_argument("--plot", dest="plot", action="store_true", default=False,
                        help="Plot the initial conditions.")
    parser.add_argument("--name", dest="run_name", default=None,
                        help="Name the run.")
    parser.add_argument("-n","--ngrid", dest="ngrid", default=128, type=int,
                        help="Number of grid points.")

    parser.add_argument("--maxxy", dest="max_xy", type=float, default=3.,
                        help="Max. abs. value of x,y")
    parser.add_argument("--q", dest="q", type=float, required=True,
                        help="q, binary mass ratio")
    parser.add_argument("--ecc", dest="ecc", type=float, default=0.,
                        help="eccentricity")
    parser.add_argument("--nu", dest="nu", type=float, default=0.,
                        help="viscosity")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    main(args.ngrid, max_xy=args.max_xy,
         q=args.q, ecc=args.ecc, nu=args.nu,
         run_name=args.run_name, plot=args.plot,
         output_path=args.output_path, overwrite=args.overwrite)

    sys.exit(0)
