# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" DOP853 integration in Cython. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.stdio cimport printf

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil

cdef extern from "dopri/dop853.h":
    ctypedef void (*GradFn)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f, GradFn gradfunc, double *gpars, unsigned norbits) nogil
    double contd8 (unsigned ii, double x)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, GradFn gradfunc, double *gpars, unsigned norbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   GradFn func, double *pars, unsigned norbits)
    double six_norm (double *x)

cdef extern from "src/_r3bp.h":
    void r3bp_derivs(unsigned ndim, double t, double *w, double *f,
                     GradFn func, double *pars, unsigned norbits)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cdef void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn):
    # TODO: see here for example in FORTRAN:
    #   http://www.unige.ch/~hairer/prog/nonstiff/dr_dop853.f
    pass

cdef void gradfn_void(double t, double *pars, double *q, double *grad):
    pass

# double dt0, int nsteps, double t0,
cpdef dop853_integrate_r3bp(double[:,::1] w0, double[::1] t,
                            double q, double ecc, double nu,
                            double atol, double rtol, int nmax):
    """

    Parameters
    ----------
    q : mass ratio
    ecc : eccentricity
    nu : viscocity
    """

    cdef:
        int i, j, k
        int res, iout
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]

        # define full array of times
        int nsteps = len(t)
        double dt0 = t[1]-t[0]
        double[::1] w = np.empty(norbits*ndim)
        double[::1] pars = np.zeros(3)

        # Note: icont not needed because nrdens == ndim
        double[:,:,::1] all_w = np.empty((nsteps,norbits,ndim))

    # store initial conditions
    for i in range(norbits):
        for k in range(ndim):
            w[i*ndim + k] = w0[i,k]
            all_w[0,i,k] = w0[i,k]

    # TODO: dense output?
    iout = 0  # no solout calls

    pars[0] = q
    pars[1] = ecc
    pars[2] = nu

    for j in range(1,nsteps,1):
        res = dop853(ndim*norbits, <FcnEqDiff> r3bp_derivs,
                     <GradFn>gradfn_void, &(pars[0]), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, solout, iout,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        for i in range(norbits):
            for k in range(ndim):
                all_w[j,i,k] = w[i*ndim + k]

    return np.asarray(t), np.asarray(all_w)
