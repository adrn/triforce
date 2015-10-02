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
from triforce.r3bp import r3bp_potential

"""
For q = 0.01

(below critical)
CJ = 3.1

(critical)
CJ = 3.15345

(above critical)
CJ = 3.2069
"""

def main(ngrid, CJ, q, ecc=0., nu=0., max_xy=3., output_path=None,
         run_name=None, overwrite=False, plot=False):
    """
    """

    # create path
    if output_path is None:
        output_path = os.path.join(project_path, "output")

        if run_name is None:
            run_name = "n{}_cj{:.3f}_q{:.0e}_e{:.0e}_nu{:.0e}".format(ngrid, CJ, q, ecc, nu)
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
        grid = np.linspace(-max_xy,max_xy,ngrid)
        z = np.zeros(ngrid*ngrid)
        xyz = np.vstack(map(np.ravel, np.meshgrid(grid, grid))+[z]).T.copy()
        U = r3bp_potential(xyz, q, ecc, nu)

        # ignore all points above ZVC
        ix = 2*U >= CJ
        xyz = xyz[ix]
        U = U[ix]

        r = np.sqrt(np.sum(xyz**2, axis=-1))
        vmag = np.sqrt(2*U - CJ)
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

    parser.add_argument("--cj", dest="jacobi_energy", type=float, required=True,
                        help="Jacobi energy")
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

    main(args.ngrid, args.jacobi_energy,
         q=args.q, ecc=args.ecc, nu=args.nu,
         run_name=args.run_name, plot=args.plot,
         output_path=args.output_path, overwrite=args.overwrite)

    sys.exit(0)
