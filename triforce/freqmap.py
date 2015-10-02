# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.integrate as gi
import gary.coordinates as gc
import gary.dynamics as gd
from superfreq import SuperFreq

# Project
from triforce.orbitgridexperiment import NoPotentialOrbitGridExperiment
from triforce.r3bp import dop853_integrate_r3bp, r3bp_potential

__all__ = ['Freqmap']

class Freqmap(NoPotentialOrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit.",
        2: "Energy conservation criteria not met.",
        3: "SuperFreq failed on find_fundamental_frequencies().",
        4: "Unexpected failure."
    }

    cache_dtype = [
        ('freqs','f8',(2,2)), # three fundamental frequencies computed in 2 windows
        ('amps','f8',(2,2)), # amplitudes of frequencies in time series
        ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
        ('success','b1'), # whether computing the frequencies succeeded or not
        ('dt','f8'), # timestep used for integration
        ('nsteps','i8'), # number of steps integrated
        ('error_code','i8') # if not successful, why did it fail? see below
    ]

    _run_kwargs = ['nperiods', 'nsteps_per_period', 'hamming_p', 'energy_tolerance',
                   'force_cartesian', 'nintvec', 'q', 'ecc', 'nu']
    config_defaults = dict(
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nperiods=256, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        hamming_p=4, # Exponent to use for Hamming filter in SuperFreq
        nintvec=15, # maximum number of integer vectors to use in SuperFreq
        force_cartesian=False, # Do frequency analysis on cartesian coordinates
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='freqmap.npy', # Name of the cache file
        q=None, # No default - must be specified
        ecc=0.,
        nu=0.
    )

    @classmethod
    def run(cls, w0, **kwargs):
        c = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                c[k] = cls.config_defaults[k]
            else:
                c[k] = kwargs[k]

        # return dict
        result = dict()

        # get timestep and nsteps for integration
        binary_period = 2*np.pi # Omega = 1
        dt = binary_period / c['nsteps_per_period']
        nsteps = c['nperiods'] * c['nsteps_per_period']

        # integrate orbit
        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            t = np.linspace(0., c['nperiods']*binary_period, nsteps)
            t,ws = dop853_integrate_r3bp(np.atleast_2d(w0).copy(), t,
                                         c['q'], c['ecc'], c['nu'],
                                         atol=1E-11, rtol=1E-10, nmax=0)
        except RuntimeError: # ODE integration failed
            logger.warning("Orbit integration failed.")
            dEmax = 1E10
        else:
            logger.debug('Orbit integrated successfully, checking energy conservation...')

            # check Jacobi energy conservation for the orbit
            E = 2*r3bp_potential(ws[:,0,:3].copy(), c['q'], c['ecc'], c['nu']) \
                - (ws[:,0,3]**2 + ws[:,0,4]**2 + ws[:,0,5]**2)
            dE = np.abs(E[1:] - E[0])
            dEmax = dE.max() / np.abs(E[0])
            logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

        if dEmax > c['energy_tolerance']:
            logger.warning("Failed due to energy conservation check.")
            result['freqs'] = np.ones((2,2))*np.nan
            result['success'] = False
            result['error_code'] = 2
            result['dE_max'] = dEmax
            return result

        # start finding the frequencies -- do first half then second half
        sf1 = SuperFreq(t[:nsteps//2+1], p=c['hamming_p'])
        sf2 = SuperFreq(t[nsteps//2:], p=c['hamming_p'])

        # define slices for first and second parts
        sl1 = slice(None,nsteps//2+1)
        sl2 = slice(nsteps//2,None)

        # TODO: the 2's below should change if we do the full 3d problem
        if c['force_cartesian']:
            fs1 = [(ws[sl1,0,j] + 1j*ws[sl1,0,j+3]) for j in range(2)]
            fs2 = [(ws[sl2,0,j] + 1j*ws[sl2,0,j+3]) for j in range(2)]

        else: # use Poincare polars
            # first need to flip coordinates so that circulation is around z axis
            new_ws = gc.cartesian_to_poincare_polar(ws)
            fs1 = [(new_ws[sl1,j] + 1j*new_ws[sl1,j+3]) for j in range(2)]
            fs2 = [(new_ws[sl2,j] + 1j*new_ws[sl2,j+3]) for j in range(2)]

        logger.debug("Running SuperFreq on the orbits")
        try:
            freqs1,d1,ixs1 = sf1.find_fundamental_frequencies(fs1, nintvec=c['nintvec'])
            freqs2,d2,ixs2 = sf2.find_fundamental_frequencies(fs2, nintvec=c['nintvec'])
        except:
            result['freqs'] = np.ones((2,2))*np.nan
            result['success'] = False
            result['error_code'] = 3
            return result

        result['freqs'] = np.vstack((freqs1, freqs2))
        result['dE_max'] = dEmax
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['amps'] = np.vstack((d1['|A|'][ixs1], d2['|A|'][ixs2]))
        result['success'] = True
        result['error_code'] = 0
        return result
