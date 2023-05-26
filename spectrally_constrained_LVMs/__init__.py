# Copyright 2023-present Ryan Balshaw
"""Spectrally-constrained-LVMs. Train linear LVMs with the addition
 of a spectral constraint with minimal effort."""
from .cost_functions import NegentropyCost, SympyCost, UserCost, VarianceCost
from .helper_methods import (
    BatchSampler,
    DataProcessor,
    DeflationOrthogonalisation,
    QuasiNewton,
    hankel_matrix,
)
from .negen_approx import CubeObject, ExpObject, LogcoshObject, QuadObject
from .spectral_constraint import SpectralObjective
from .spectrally_constrained_model import LinearModel

__author__ = "Ryan Balshaw"
__version__ = "0.1.1"
__email__ = "ryanbalshaw81@gmail.com"
__description__ = (
    "Train linear LVMs with the addition "
    "of a spectral constraint with minimal effort."
)
# __uri__ = "http://spectrally-constrained-lvms.readthedocs.io/"
__all__ = [
    "SpectralObjective",
    "NegentropyCost",
    "SympyCost",
    "UserCost",
    "VarianceCost",
    "hankel_matrix",
    "LinearModel",
    "BatchSampler",
    "DataProcessor",
    "DeflationOrthogonalisation",
    "QuasiNewton",
    "CubeObject",
    "ExpObject",
    "LogcoshObject",
    "QuadObject",
]
