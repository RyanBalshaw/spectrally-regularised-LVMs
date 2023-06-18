=========
Tutorials
=========

In this section, different tutorials are provided on how to best make use of the package.

Tutorial 1: Using an LVM
========================

To use spectrally-regularised-LVMs, users will need to perform each of the seven steps given in the code block below.

.. code-block:: python

    # Define imports
    import spectrally_regularised_LVMs as srLVMs
    np.random.seed(0)

    # Step 1: Load in the time series signal
    Fs = 20480

    data_dict = np.loadtxt("./2004.02.16.04.32.39") # This is the IMS dataset signal stored in the '/Examples/' directory on the Github page
    signal_data = data_dict[:, 0]

    # Step 2: Perform pre-processing
    Lw = 256
    Lsft = 1
    X = srLVMs.hankel_matrix(signal_data, Lw, Lsft) # Hankelise the vibration data

    # Step 3: Define the cost function
    cost_inst = srLVMs.NegentropyCost("exp", {"alpha":1}) # negentropy objective

    # Step 4: Define the model
    model_inst = srLVMs.LinearModel(n_sources = 5,
                                    cost_instance = cost_inst,
                                    organise_by_kurt = True,
                                    alpha_reg = 1.0,
                                    verbose = True)

    # Step 5: Estimate the model parameters
    model_inst.fit(X, n_iters = 500, learning_rate = 1, tol = 1e-4, Fs = Fs)

    # Step 6: Obtain the latent representation Z
    Z = model_inst.transform(X)

    # Step 7: Obtain the recovered representation of X
    X_recon = model_inst.inverse_transform(Z)

Tutorial 2: Using the default objective functions
=================================================

Two common LVM objectives are provided for users to get up and running with the package. Note that although the objective functions are to be maximised, the implementation strategy followed in this package performs minimisation. Converting an objective function which is to be maximised into one for minimisation simply results in changing the sign of the objective function.

Negentropy objective function:
------------------------------

.. code-block:: python

    # Define imports
    import spectrally_regularised_LVMs as srLVMs
    import numpy as np
    np.random.seed(0)

    # Setup general X matrix
    X_ = np.random.randn(1000, 16)
    w_ = np.random.randn(16, 1)
    y_ = X_ @ w_

    # Initialise the cost function instance
    cost_inst = srLVMs.NegentropyCost(source_name="exp",
                                      source_params={"alpha": 1})

    # Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X_, w_, y_, 1e-4)
    res_hess = cost_inst.check_hessian(X_, w_, y_, 1e-4)


Variance objective function:
----------------------------

.. code-block:: python

    # Define imports
    import spectrally_regularised_LVMs as srLVMs
    import numpy as np
    np.random.seed(0)

    # Initialise the cost function instance
    cost_inst = srLVMs.VarianceCost(use_hessian=True, verbose=True)

    # Check that the gradient and Hessian make sense
    res_grad = cost_inst.check_gradient(X_, w_, y_, 1e-4)
    res_hess = cost_inst.check_hessian(X_, w_, y_, 1e-4)

Tutorial 3: Defining your own objective function
================================================

This package allows users to implement their own objective functions. Two examples are shown here, one example where the user implements their objective function and associated derivatives from scratch, and one where `Sympy <https://www.sympy.org/en/index.html>`_ is used to obtain the necessary derivatives.

Method one: User defined objective
----------------------------------

This method allows users to implement their objective function and all required higher order derivatives manually. This is demonstrated through:

.. code-block:: python

    import numpy as np
    import spectrally_regularised_LVMs as srLVMs

    # Define objective function (maximise source variance)
    def cost(X, w, y):

        return -1 * np.mean((X @ w) ** 2, axis=0) # Framework performs minimisation

    # Define gradient vector
    def grad(X, w, y):

        return -2 * np.mean(y * X, axis=0, keepdims=True).T

    # Define Hessian matrix
    def hess(X, w, y):

        return -2 * np.cov(X, rowvar=False)

    # Initialise the cost instance
    user_cost = srLVMs.UserCost(use_hessian = True)

    # Define the objective function, gradient and Hessian
    user_cost.set_cost(cost)
    user_cost.set_gradient(grad)
    user_cost.set_hessian(hess)

    # Check the implementation
    X_ = np.random.randn(1000, 16)
    w_ = np.random.randn(16, 1)
    y_ = X_ @ w_

    res_grad = user_cost.check_gradient(X_, w_, y_,step_size = 1e-4)
    res_hess = user_cost.check_hessian(X_, w_, y_,step_size = 1e-4)

Method two: SymPy defined objective
-----------------------------------

Users can also use `Sympy <https://www.sympy.org/en/index.html>`_ to implement their objective function, which allows for all higher order derivatives to be obtained symbolically. An example of this is given through

.. code-block:: python

    import sympy as sp
    import numpy as np
    import spectrally_regularised_LVMs as srLVMs

    n_samples = 1000 # Fix the number of samples in the data
    n_features = 16 # Fix the number of features

    # Initialise the cost function instance
    user_cost = srLVMs.SympyCost(n_samples, n_features, use_hessian=True)

    # Get the SymPy representations of the model parameters
    X_sp, w_sp, iter_params = user_cost.get_model_parameters()
    i, j = iter_params

    # Calculate the objective function (maximise source variance)
    loss_i = sp.Sum(w_sp[j, 0] * X_sp[i, j], (j, 0, n_features - 1))
    loss = -1 / n_samples * sp.Sum(loss_i**2, (i, 0, n_samples - 1))

    # Set the properties within the instance
    user_cost.set_cost(loss)

    # Use SymPy to calculate the first and second order derivatives
    user_cost.implement_methods()

    # Check the implementation
    X_ = np.random.randn(n_samples, n_features)
    w_ = np.random.randn(n_features, 1)
    y_ = X_ @ w_

    res_grad = user_cost.check_gradient(X_, w_, y_,step_size = 1e-4)
    res_hess = user_cost.check_hessian(X_, w_, y_,step_size = 1e-4)
