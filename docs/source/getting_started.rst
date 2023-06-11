Getting started
===============

Time-series data hankelisation
------------------------------

The ``spectrally-regularised-lvms.hankel_matrix()`` function can be used to convert a signal into a hankel matrix. A simple example of this can be given for some sampled representation of a sinusoidal function.

>>> import spectrally_regularised_LVMs as srLVMs
>>> import numpy as np
>>> x_signal = np.sin(2 * np.pi * np.arange(0, 10, 1/1000))
>>> X = srLVMs.hankel_matrix(x_signal, Lw=256, Lsft=1)

Defining an objective function
------------------------------

The simplest way to define an objective function is to use one of the provided variance maximisation or negentropy maximisation objective functions.

Variance maximisation
_____________________

>>> import spectrally_regularised_LVMs as srLVMs
>>> PCA_objective = srLVMs.VarianceCost(use_hessian=True, verbose=True)

Negentropy maximisation
_____________________

>>> import spectrally_regularised_LVMs as srLVMs
>>> PCA_objective = srLVMs.NegentropyCost(source_name="exp", source_params={"alpha": 1})

Creating the LVM instance
-----------------------------

Initialising the LVM instance is
