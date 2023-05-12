# Copyright 2023-present Ryan Balshaw
"""Spectrally-constrained-LVMs. Train linear LVMs with the addition
 of a spectral constraint with minimal effort."""
from .constraint import spectral_objective
from .cost_functions import negentropy_cost, sympy_cost, user_cost
from .helper import (
    Hankel_matrix,
    batch_sampler,
    data_processor,
    deflation_orthogonalisation,
    quasi_Newton,
)
from .negen_approx import cube_object, exp_object, logcosh_object, quad_object
from .spectrally_constrained_linear_model import linear_model

__author__ = "Ryan Balshaw"
__version__ = "0.1.0"
__email__ = "ryanbalshaw81@gmail.com"
__description__ = (
    "Train linear LVMs with the addition "
    "of a spectral constraint with minimal effort."
)
__uri__ = ""
__all__ = [
    "spectral_objective",
    "negentropy_cost",
    "sympy_cost",
    "user_cost",
    "Hankel_matrix",
    "batch_sampler",
    "data_processor",
    "deflation_orthogonalisation",
    "quasi_Newton",
    "cube_object",
    "exp_object",
    "logcosh_object",
    "quad_object",
    "linear_model",
]
