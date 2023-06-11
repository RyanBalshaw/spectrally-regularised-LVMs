Package guide
=============

Documentation is included in the modules that make up the spectrally-regularised-LVMs package. Each script contains classes and functions for handling each aspect needed for the latent variable model  (LVM) utilisation pipeline.

For beginners in programming, it is advised to look at the example applications provided in the `GitHub repository <https://github.com/RyanBalshaw/spectrally-regularised-LVMs>`_ and adapt them accordingly for your own use.

negen_approx
------------

This module handles the negentropy approximation functions used for negentropy-based ICA methodologies. There are four approximation functions which are implemented in this module, namely: 1) the :py:class:`cube <spectrally_regularised_LVMs.negen_approx.CubeObject>` objective where :math:`G(u) = u^3`, 2) the :py:class:`exp <spectrally_regularised_LVMs.negen_approx.ExpObject>` objective where :math:`G(u, a_2) = \frac{1}{a_2} \exp \left( -a_2 \cdot \frac{u^2}{2} \right)`, 3) the :py:class:`logcosh <spectrally_regularised_LVMs.negen_approx.LogcoshObject>` objective where :math:`G(u, a_1) = \frac{1}{a_1} \cdot \log \left( \frac{\exp(a_1 \cdot u) + \exp(-a_1 \cdot u)}{2} \right)`, and 4) the :py:class:`quad <spectrally_regularised_LVMs.negen_approx.QuadObject>` objective where :math:`G(u) = \frac{1}{4} \cdot u^4`.


cost_functions
--------------

This module handles the objective function methods used for LVM parameter estimation. It provides the objective functions for two common methodologies, principal component analysis (PCA) and negentropy-based independent component analysis (ICA), and then gives two additional classes that allow users to implement their own objective functions. For the latter, one class allows users to implement their objective function and all associated derivatives as methods that return NumPy matrices, and another which allows users to symbolically implement their objective function, and then all derivatives are obtained symbolically without any user implementation requirements.


spectral_regulariser
--------------------

This module handles the spectral regularisation term that penalises non-orthogonality between the spectral representations of the associated source vectors :math:`\mathbf{w}`. The objective of this regularisation term is to ensure that the source vectors do not capture duplicate information in the single channel data. The module implements the objective function and its gradient vector and hessian matrix based off two vectors :math:`\mathbf{w}_i` and :math:`\mathbf{w}_j`, where :math:`\mathbf{w}_i` represents the source vector that is being adjusted to minimise the user's objective function, and the :math:`\mathbf{w}_j` vector represents the previously learnt source vector that we wish to enforce that :math:`\mathbf{w}_i` is orthogonal to.


helper_methods
--------------

This module handles any helper functions for the LVM parameter estimation step. A :py:func:`data hankelisation <spectrally_regularised_LVMs.helper_methods.hankel_matrix>` is provided to perform time-series signal hankelisation. A :py:class:`batch sampler <spectrally_regularised_LVMs.helper_methods.BatchSampler>` class is provided to assist with any batch sampling steps that may be required by the user. A :py:class:`data pre-processor <spectrally_regularised_LVMs.helper_methods.DataProcessor>` class is provided to implement any data pre-processing steps that are required to operate on the data matrix :math:`\mathbf{X}`. A :py:class:`deflation orthogonalisation <spectrally_regularised_LVMs.helper_methods.DeflationOrthogonalisation>` class is provided which performs Gram-Schmidt orthogonalisation. Finally, a :mod:`quasi-Newton <spectrally_regularised_LVMs.helper_methods.QuasiNewton>` class is provided to perform any quasi-Newton Hessian approximation strategies that the user may wish to use.

spectrally_regularised_model
----------------------------

This module handles the LVM initialisation and subsequent model parameter estimation step. The :py:class:`LinearModel spectrally_regularised_LVMs.spectrally_regularised_model.LinearModel` class allows users to initialise an instance of this class, and this instance represents the LVM prior to model parameter estimation. This class handles all choices made by the user regarding parameter estimation and the optimisation methodology used, estimates the model parameters, and performs the transition to and from the latent space. In this way, it encapsulates the LVM as a Python class instance.
