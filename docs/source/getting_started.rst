===============
Getting started
===============

Time-series data hankelisation
==============================

The ``spectrally_regularised_lvms.hankel_matrix()`` function can be used to convert a single channel time-series signal into a hankel matrix. The hankel matrix of a signal is given by

.. math::

    \mathbf{X} =
    \begin{bmatrix}
        x[1] & \cdots & x[L_w] \\
		x[L_{sft}] & \cdots & x[L_{sft} + L_w] \\
		x[2 \cdot L_{sft}] & \cdots & x[2 \cdot L_{sft} + L_w] \\
		\vdots & \ddots  & \vdots \\
		x[L_{sft} \cdot (L_H - 1)] & \cdots & x[L_{sft} \cdot (L_H - 1) + L_w]
    \end{bmatrix}

where :math:`L_w` is the window length, :math:`L_{sft}` is the shift parameter, and :math:`L_H = \lfloor\frac{L - L_w}{L_{sft}}\rfloor + 1` represents the number of rows in :math:`\mathbf{X} \in \mathbb{R}^{L_H \times L_w}`. A simple example of this can be given for some sampled representation of a sinusoidal function.

.. code-block:: python
    :linenos:

    import spectrally_regularised_LVMs as srLVMs
    import numpy as np

    Fs = 1000 # Sampling frequency
    t = np.arange(0, 10, 1/Fs) # Time representation of the signal
    x_signal = np.sin(2 * np.pi * t) # Create a toy signal
    X = srLVMs.hankel_matrix(x_signal, Lw=256, Lsft=1) # Develop the hankel matrix

Defining an objective function
==============================

The simplest way to define an objective function/cost instance is to use one of the provided variance maximisation or negentropy maximisation objective functions. Alternatively, a symbolic representation of the objective function can be used, or an explicit representation can be given via user-defined functions.

To ensure that any gradient or Hessian information can be computed for any objective function, a ``finite_diff_flag`` is given to each of the class instances that are used for objective function definition. This ensures that an approximation of the gradient or Hessian information can be readily computed if required.

.. note::
    Although the objective functions used in this example is required to be maximised, the implementation strategy followed in this package performs minimisation. Converting an objective function which is to be maximised into one for minimisation simply results in changing the sign of the objective function.

For the symbolic and explicit objective functions, the principal component analysis (PCA) objective function is demonstrated here. This is given by

.. math::

    \begin{align}
    \mathcal{L}_{model}(\mathbf{w}_i) &= - \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} \{ \left(\mathbf{w}_i^T \mathbf{x} \right)^2 \} \\
    &= - \frac{1}{N} \sum_{j=1}^{N} z_j^2,
    \end{align}

where it is assumed that :math:`\mathbf{x}` is zero-mean and :math:`z_j = \mathbf{w}_i^T \mathbf{x}_j`. In a symbolic representation, the indices of :math:`z_j` are symbolically represented for :math:`N` samples from :math:`p(\mathbf{x})`.

Symbolic objective functions
----------------------------

For the symbolic implementation of an objective function, `SymPy <https://www.sympy.org/en/index.html>`_ is used. It was decided that the easiest way to allow users to implement different objective functions is to create an indexable :math:`z` variable, with index :math:`j`. This ensures that users can define a simple form of their objective function without having to create variables for the LVM component vectors :math:`\mathbf{w}` and the hankel matrix :math:`\mathbf{X}`. The index :math:`j` is configured to have a range of :math:`j \in [0, n-1]` where :math:`n` refers to the samples of the data feature vectors :math:`\mathbf{x}_j`. The gradient and hessian information is then obtained by symbolically deriving the objective function and exploiting the properties of the linear LVM transformation :math:`z_j = \mathbf{w}_i^T \mathbf{x}_j`.

.. code-block:: python
    :linenos:

    import sympy as sp
    import spectrally_regularised_LVMs as srLVMs

    # Symbolic cost function implementation
    cost_inst = srLVMs.SymbolicCost(use_hessian = True,
                                    verbose = False,
                                    finite_diff_flag = False)

    z, j, n = cost_inst.get_symbolic_parameters()

    loss = -1/n * sp.Sum((z[j])**2, (j))

    cost_inst.set_cost(loss)

    # Visualise the loss
    sp.pretty_print(loss)

    # Visualise the properties of the indexed variables
    print(z[j].shape, z[j].ranges)
    print(j)
    print(n)

Explicit objective functions
----------------------------

Alternatively, users can provide an explicit version of an objective function via functions that use NumPy objects. This was done to allow for cases where specific objective functions, gradient vectors, or Hessian matrices can be encoded by the user. These functions expect three inputs: the hankel matrix :math:`\mathbf{X}`, the component vector :math:`\mathbf{w}`, and the latent transformation vector :math:`\mathbf{z} = \mathbf{X} \mathbf{w}`. Using the ``finite_diff_flag = True`` implies that the gradient and Hessian are approximated, and hence there would be no need to supply the gradient or Hessian functions.

.. code-block:: python
    :linenos:

    import numpy as np
    import spectrally_regularised_LVMs as srLVMs

    # Explicit cost function implementation
    cost_inst = srLVMs.ExplicitCost(use_hessian = True,
                                    verbose = False,
                                    finite_diff_flag = False)

    obj = lambda X, w, z: -1 * np.mean((X @ w)**2, axis = 0)
    grad = lambda X, w, z: -2 * np.mean(z * X, axis = 0,
                                        keepdims=True).T
    hess = lambda X, w, z: -2 / X.shape[0] * (X.T @ X)

    # Set the properties
    cost_inst.set_cost(obj)
    cost_inst.set_gradient(grad)
    cost_inst.set_hessian(hess)

Variance maximisation
---------------------

The variance maximisation objective tries to obtain orthogonal projections wherein the variance of the projected samples is maximal. This objective is implemented as it is a common objective in the literature. The code to use this objective is given below.

.. code-block:: python
    :linenos:

    PCA_objective = srLVMs.VarianceCost(use_hessian = True,
                                        verbose = False,
                                        finite_diff_flag = False)

Negentropy maximisation
-----------------------

The negentropy maximisation objective tries to obtain projections that maximise the non-Gaussianity of the latent sources. The code to use this objective is given below.

.. code-block:: python
    :linenos:

    ICA_objective = srLVMs.NegentropyCost(source_name="exp",
                                          source_params={"alpha": 1},
                                          use_hessian = True,
                                          verbose = False,
                                          finite_diff_flag = False)

Initialising the LVM
====================

The LVM can be initialised using an instance of the  ``spectrally_regularised_lvms.LinearModel()`` class. Please refer to the :ref:`class documentation <modules:Spectrally regularised model>` or the documentation :ref:`argument tables <target to guides>` for more information regarding the parameters of this class.

.. code-block:: python
    :linenos:

    model_inst = srLVMs.LinearModel(n_sources = ...,
                                cost_instance = cost_inst,
                                Lw = ...,
                                Lsft = ...,
                                verbose = True)

where ``n_sources`` represents the number of latent source components (:math:`d`) that are to be estimated, :math:`L_w` represents the window length the signal segments stored in the Hankel matrix, and :math:`L_{sft}` represents the shift parameter used to develop the hankel matrix.

Estimating the LVM parameters
=============================

Estimating the model parameters is achieved using the ``LinearModel().fit(...)`` method.

.. code-block:: python
    :linenos:

    model_inst = model_inst.fit(x_signal,
                                n_iters = 500,
                                learning_rate = 1,
                                tol = 1e-4,
                                Fs = Fs)

where ``x_signal`` refers to the single channel time-series signal. The arguments for the ``.fit(.)`` call are detailed in the documentation :ref:`argument tables <target to guides>`.

Using the LVM
=============

Using the LVM reduces to using the ``.transform(...)`` method and the ``.inverse_transform()`` method for an instance of the ``LinearModel`` class.

.. code-block:: python
    :linenos:

    # Transition to the latent space
    Z_latent = model_inst.transform(x_signal)

    # Transition back to the data space
    X_recon = model_inst.inverse_transform(Z_latent)

Note that ``X_recon`` is the reconstructed hankel matrix of :math:`\mathbf{X}`. This concludes the code to get started with the package.
