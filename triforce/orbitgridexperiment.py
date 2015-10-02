# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
try:
    import cPickle as pickle
except ImportError: # only works in Python 3
    import pickle

try:
    from abc import abstractclassmethod
except ImportError: # only works in Python 3
    class abstractclassmethod(classmethod):

        __isabstractmethod__ = True

        def __init__(self, callable):
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)

# Third-party
from astropy import log as logger
import numpy as np

# Project
from streammorphology.experimentrunner import OrbitGridExperiment

class NoPotentialOrbitGridExperiment(OrbitGridExperiment):

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # read out just this initial condition
        norbits = len(self.w0)
        allfreqs = np.memmap(self.cache_file, mode='r',
                             shape=(norbits,), dtype=self.cache_dtype)

        # short-circuit if this orbit is already done
        if allfreqs['success'][index]:
            logger.debug("Orbit {0} already successfully completed.".format(index))
            return None

        # Only pass in things specified in _run_kwargs (w0 required)
        kwargs = dict([(k,self.config[k]) for k in self.config.keys() if k in self._run_kwargs])
        res = self.run(w0=self.w0[index], **kwargs)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'w') as f:
            pickle.dump(res, f)
        return tmpfile

    @abstractclassmethod
    def run(cls, w0, **kwargs):
        """ (classmethod) Run the experiment on a single orbit """
