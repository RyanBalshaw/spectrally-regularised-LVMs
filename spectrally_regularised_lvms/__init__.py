"""
MIT License
-----------

Copyright (c) 2023 Ryan Balshaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------

Spectrally-regularised-LVMs. Train linear LVMs with the addition of a spectral
regularisation term with minimal effort.
"""
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
__version__ = "0.0.0"
__email__ = "ryanbalshaw81@gmail.com"
__description__ = "A framework of linear LVMs with spectral regularisation."
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
