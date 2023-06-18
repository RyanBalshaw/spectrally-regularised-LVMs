===============
Getting started
===============

Time-series data hankelisation
==============================

The ``spectrally_regularised_lvms.hankel_matrix()`` function can be used to convert a signal into a hankel matrix. A simple example of this can be given for some sampled representation of a sinusoidal function.

.. code-block:: python
    :linenos:

    import spectrally_regularised_LVMs as srLVMs
    import numpy as np

    Fs = 1000
    t = np.arange(0, 10, 1/Fs)
    x_signal = np.sin(2 * np.pi * t)
    X = srLVMs.hankel_matrix(x_signal, Lw=256, Lsft=1)

Defining an objective function
==============================

The simplest way to define an objective function is to use one of the provided variance maximisation or negentropy maximisation objective functions. Note that although the objective functions are to be maximised, the implementation strategy followed in this package performs minimisation. Converting an objective function which is to be maximised into one for minimisation simply results in changing the sign of the objective function.

Variance maximisation
---------------------

The variance maximisation objective tries to obtain orthogonal projections wherein the variace of the projected samples is maximal.

.. code-block:: python
    :linenos:

    PCA_objective = srLVMs.VarianceCost(use_hessian=True, verbose=True)

Negentropy maximisation
-----------------------

The negentropy maximisation

.. code-block:: python
    :linenos:

    ICA_objective = srLVMs.NegentropyCost(source_name="exp", source_params={"alpha": 1})

Initialising the LVM
====================

The LVM can be initialised using an instance of the  ``spectrally_regularised_lvms.LinearModel()`` class. Please refer to the :ref:`class documentation <modules:Spectrally regularised model>` for more information regarding the parameters of this class.

.. code-block:: python
    :linenos:

    model_inst = srLVMs.LinearModel(n_sources = 5,
                                    cost_instance = ICA_objective,
                                    organise_by_kurt = True,
                                    alpha_reg = 1.0,
                                    verbose = True)




Estimating the LVM parameters
=============================

Estimating the model parameters is achieved using the ``LinearModel().fit(...)`` method.

.. code-block:: python
    :linenos:

    model_inst = model_inst.fit(X,
                                n_iters = 500,
                                learning_rate = 1,
                                tol = 1e-4,
                                Fs = Fs)

Using the LVM
=============

Using the LVM reduces to using the ``LinearModel().transform(...)`` method and the ``LinearModel().inverse_transform()`` method.

.. code-block:: python
    :linenos:

    # Transition to the latent space
    Z_latent = model_inst.transform(X)

    # Transition back to the data space
    X_recon = model_inst.inverse_transform(X)

This concludes the code to get started with the package.
