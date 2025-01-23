=========
Tutorials
=========

In this section, different tutorials are provided on how to best make use of the package.

Tutorial 1: Using a LVM
=======================

To use spectrally-regularised-LVMs, users will need to perform each of the six steps given in the code block below.

.. code-block:: python

    # Define imports
    import numpy as np
    import spectrally_regularised_lvms as srLVMs
    np.random.seed(0)

    # Step 1: Load in the time series signal
    data_dict = np.loadtxt("./2004.02.17.07.12.39") # This is the IMS dataset signal stored in the '/Examples/' directory on the Github page
    signal_data = data_dict[:, 0]
    Fs = 20480

    # Step 2: Define the cost function instance
    cost_inst = srLVMs.NegentropyCost("exp", {"alpha":1}) # negentropy objective

    # Step 3: Define the model
    model_inst = srLVMs.LinearModel(n_sources = 10,
                                    cost_instance = cost_inst,
                                    Lw = 256,
                                    Lsft = 1,
                                    sumt_flag=True,
                                    verbose = True)

    # Step 4: Estimate the model parameters
    model_inst.fit(x_signal)

    # Step 5: Obtain the latent representation Z
    Z = model_inst.transform(x_signal)

    # Step 6: Obtain the recovered representation of X
    X_recon = model_inst.inverse_transform(Z)

Tutorial 2: Defining your own objective function
================================================

This package allows users to implement their own objective functions. Two examples are shown here, one example where the user implements their objective function and associated derivatives from scratch, and one where `Sympy <https://www.sympy.org/en/index.html>`_ is used to obtain the necessary derivatives.

Method one: Symbolically defined objective
------------------------------------------

Users can use `Sympy <https://www.sympy.org/en/index.html>`_ to implement their objective function, which allows for all higher order derivatives to be obtained symbolically. An example of this is given through

.. code-block:: python

    # Define imports
    import spectrally_regularised_lvms as srLVMs
    import numpy as np
    import sympy as sp
    np.random.seed(0)

    # Setup general matrices
    X = np.random.randn(500, 16)
    X -= np.mean(X, axis = 0, keepdims=True)
    w = np.random.randn(16, 1)
    z = X @ w

    # Initialise the cost function instance
    cost_inst = srLVMs.SymbolicCost(use_hessian=True,
                                    verbose=True,
                                    finite_diff_flag=False)

    z_sp, j, n = cost_inst.get_symbolic_parameters()

    loss = -1/n * sp.Sum((z_sp[j])**2, (j))

    cost_inst.set_cost(loss)
    display(loss)

    # Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X, w, z, 1e-4)
    res_hess = cost_inst.check_hessian(X, w, z, 1e-4)

Method two: Explicitly defined objective
----------------------------------------

This method allows users to implement their objective function and all required higher order derivatives manually. This is demonstrated through the PCA objective function:

.. code-block:: python

    # Define imports
    import spectrally_regularised_lvms as srLVMs
    import numpy as np
    np.random.seed(0)

    # Setup general X matrix
    X = np.random.randn(500, 16)
    X -= np.mean(X, axis = 0, keepdims=True)
    w = np.random.randn(16, 1)
    z = X @ w

    # Initialise the cost function instance
    cost_inst = srLVMs.ExplicitCost(use_hessian=True,
                                    verbose=True,
                                    finite_diff_flag=False)

    # Implement the objective function, gradient and Hessian
    def loss(X, w, z):
        return -1 * np.mean((X @ w) ** 2, axis=0)

    def grad(X, w, z):
        return -2 * np.mean(z * X, axis=0, keepdims=True).T

    def hess(X, w, z):
        return -2 / X.shape[0] * (X.T @ X)

    # Set the properties
    cost_inst.set_cost(loss)
    cost_inst.set_gradient(grad)
    cost_inst.set_hessian(hess)

    # Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X, w, z, 1e-4)
    res_hess = cost_inst.check_hessian(X, w, z, 1e-4)

Tutorial 3: Using the default objective functions
=================================================

Two common LVM objectives are provided for users to get up and running with the package. Note that although the objective functions are to be maximised, the implementation strategy followed in this package performs minimisation. Converting an objective function which is to be maximised into one for minimisation simply results in changing the sign of the objective function.

Variance objective function:
----------------------------

.. code-block:: python

    # Define imports
    import spectrally_regularised_lvms as srLVMs
    import numpy as np
    np.random.seed(0)

    # Setup general matrices
    X = np.random.randn(500, 16)
    X -= np.mean(X, axis = 0, keepdims=True) # De-mean the data
    w = np.random.randn(16, 1)
    z = X @ w

    # Initialise the cost function instance
    cost_inst = srLVMs.VarianceCost(use_hessian=True,
                                    verbose=True,
                                    finite_diff_flag=False)

    # Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X, w, z, 1e-4)
    res_hess = cost_inst.check_hessian(X, w, z, 1e-4)

Negentropy objective function:
------------------------------

.. code-block:: python

    # Define imports
    import spectrally_regularised_lvms as srLVMs
    import numpy as np
    np.random.seed(0)

    # Setup general matrices
    X_ = np.random.randn(500, 16)
    X_ -= np.mean(X_, axis = 0, keepdims=True) # De-mean the data
    w_ = np.random.randn(16, 1)
    z_ = X_ @ w_

    ## Initialise the cost function instance
    cost_inst = srLVMs.NegentropyCost(source_name="exp",
                                      source_params={"alpha": 1},
                                      use_approx=False,
                                      use_hessian=True,
                                      verbose = True,
                                      finite_diff_flag=False)

    ## Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X_, w_, z_, 1e-4)
    res_hess = cost_inst.check_hessian(X_, w_, z_, 1e-4)
