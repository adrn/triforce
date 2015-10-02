# coding: utf-8

from __future__ import division, print_function

"""
Frequency map the restricted three-body problem. Before calling this module,
you'll need to generate a grid of initial conditions or make sure the grid you
have is in the correct format.

For example, you might do::

    python scripts/make_grid.py --cj=3.1 --q=0.01 -n 512

and then run this module on::

    python scripts/freq_map.py --path=output/<pathname>

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from triforce.freqmap import Freqmap

runner = ExperimentRunner(ExperimentClass=Freqmap)
runner.run()

sys.exit(0)
