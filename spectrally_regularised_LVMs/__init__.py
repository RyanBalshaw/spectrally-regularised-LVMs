# Copyright 2023-present Ryan Balshaw
"""Spectrally-regularised-LVMs. Train linear LVMs with the addition
 of a spectral regularisation term with minimal effort."""
from .cost_functions import ExplicitCost, NegentropyCost, SymbolicCost, VarianceCost
from .helper_methods import (
    BatchSampler,
    DataProcessor,
    DeflationOrthogonalisation,
    QuasiNewton,
    hankel_matrix,
)
from .negen_approx import CubeObject, ExpObject, LogcoshObject, QuadObject
from .spectral_regulariser import SpectralObjective
from .spectrally_regularised_model import LinearModel

__author__ = "Ryan Balshaw"
__version__ = "0.1.2"
__email__ = "ryanbalshaw81@gmail.com"
__description__ = (
    "Train linear LVMs with the addition "
    "of a spectral regularisation term with minimal effort."
)
__uri__ = "http://spectrally-regularised-lvms.readthedocs.io/"
__all__ = [
    "SpectralObjective",
    "NegentropyCost",
    "SymbolicCost",
    "ExplicitCost",
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
