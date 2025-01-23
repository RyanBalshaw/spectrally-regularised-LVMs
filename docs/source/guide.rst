=============
Package guide
=============

Documentation is included in the modules that make up the spectrally-regularised-LVMs package. Each script contains classes and functions for handling each aspect needed for the latent variable model  (LVM) utilisation pipeline.

For beginners in programming, it is advised to look at the example applications provided in the `GitHub repository <https://github.com/RyanBalshaw/spectrally-regularised-LVMs>`_ and adapt them accordingly for your own use.

negen_approx
============

This module handles the negentropy approximation functions used for negentropy-based ICA methodologies. There are four approximation functions which are implemented in this module, namely: 1) the :py:class:`cube <spectrally_regularised_lvms.negen_approx.CubeObject>` objective where :math:`G(u) = u^3`, 2) the :py:class:`logcosh <spectrally_regularised_lvms.negen_approx.LogcoshObject>` objective where :math:`G(u, a_1) = \frac{1}{a_1} \cdot \log \left( \frac{\exp(a_1 \cdot u) + \exp(-a_1 \cdot u)}{2} \right)`, 3) the :py:class:`exp <spectrally_regularised_lvms.negen_approx.ExpObject>` objective where :math:`G(u, a_2) = - \frac{1}{a_2} \exp \left( -a_2 \cdot \frac{u^2}{2} \right)`,  and 4) the :py:class:`quad <spectrally_regularised_lvms.negen_approx.QuadObject>` objective where :math:`G(u) = \frac{1}{4} \cdot u^4`.


cost_functions
==============

This module handles the objective function methods used for LVM parameter estimation. It provides the objective functions for two common methodologies, principal component analysis (PCA) and negentropy-based independent component analysis (ICA), and then gives two additional classes that allow users to implement their own objective functions. For the latter, one class allows users to implement their objective function and all associated derivatives as methods that use NumPy objects, and another which allows users to symbolically implement their objective function, and then all derivatives are obtained symbolically without any user implementation requirements.


spectral_regulariser
====================

This module handles the spectral regularisation term that penalises non-orthogonality between the spectral representations of the associated source vectors :math:`\mathbf{w}`. The objective of this regularisation term is to ensure that the source vectors do not capture duplicate information in the single channel data. The module implements the objective function and its gradient vector and hessian matrix based off two vectors :math:`\mathbf{w}_i` and :math:`\mathbf{w}_j`, where :math:`\mathbf{w}_i` represents the source vector that is being adjusted to minimise the user's objective function, and the :math:`\mathbf{w}_j` vector represents the previously learnt source vector that we wish to enforce that :math:`\mathbf{w}_i` is orthogonal to.


helper_methods
==============

This module handles any helper functions for the LVM parameter estimation step. A :py:func:`data hankelisation <spectrally_regularised_lvms.helper_methods.hankel_matrix>` is provided to perform time-series signal hankelisation. A :py:class:`batch sampler <spectrally_regularised_lvms.helper_methods.BatchSampler>` class is provided to assist with any batch sampling steps that may be required by the user. A :py:class:`data pre-processor <spectrally_regularised_lvms.helper_methods.DataProcessor>` class is provided to implement any data pre-processing steps that are required to operate on the data matrix :math:`\mathbf{X}`. A :py:class:`deflation orthogonalisation <spectrally_regularised_lvms.helper_methods.DeflationOrthogonalisation>` class is provided which performs Gram-Schmidt orthogonalisation. Finally, a :mod:`quasi-Newton <spectrally_regularised_lvms.helper_methods.QuasiNewton>` class is provided to perform any quasi-Newton Hessian approximation strategies that the user may wish to use.

spectrally_regularised_model
============================

This module handles the LVM initialisation and subsequent model parameter estimation step. The :py:class:`LinearModel <spectrally_regularised_lvms.spectrally_regularised_model.LinearModel>` class allows users to initialise an instance of this class, and this instance represents the LVM prior to model parameter estimation. This class handles all choices made by the user regarding parameter estimation and the optimisation methodology used, estimates the model parameters, and performs the transition to and from the latent space. In this way, it encapsulates the LVM as a Python class instance.

.. _target to guides:

.. list-table:: The arguments for instances of the ``LinearModel`` class and their associated types, descriptions, and default values.
   :widths: 10 10 70 10
   :header-rows: 1

   * - Python API
     - Type
     - Description
     - Default value
   * - ``n_sources``
     - int
     - Details the number of latent sources that are to be estimated, where :math:`1 \leq n_{sources} \leq D`.
     - Required argument
   * - ``cost_instance``
     - class instance
     - Asks for an instance of the classes from the cost\_function sub-module that inherit from the CostClass parent class.
     - Required argument
   * - ``Lw``
     - int
     - Specifies the window length used for the data hankelisation step.
     - Required argument
   * - ``Lsft``
     - int
     - Specifies the window shift parameter used for the data hankelisation step.
     - Required argument
   * - ``whiten``
     - bool
     - Specifies whether :math:`\mathbf{X}` is to be de-meaned and pre-whitened, i.e. exhibit the latent sources are uncorrelated and have unit variance, or just de-meaned.
     - True
   * - ``init_type``
     - str
     - Defines the initialisation procedure for :math:`\mathbf{W}`. The option ``init_type`` = "broadband" will ensure that :math:`w_i[n] = \delta[n]` where :math:`\delta[0]=1` and 0 otherwise, while the option ``init_type`` = "random" will randomly initialise :math:`\mathbf{w}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`.
     - "broadband"
   * - ``perform_gso``
     - bool
     - Specifies whether Gram-Schmidt orthogonalisation [1] must be performed when estimating :math:`\mathbf{w}_j,\, j > 1`.
     - True
   * - ``batch_size``
     - int | None
     - Specifies the batch size for the samples from :math:`\mathbf{x}`. If ``batch_size = None`` then the full input :math:`\mathbf{X}` is used.
     - None
   * - ``var_PCA``
     - float | None
     - Specifies whether PCA is used as a pre-processing step to reduce the dimensionality of :math:`\mathbf{X}` by discarding trailing eigenvalues. Note that this step is debated in the literature [2]. If the user wishes to use this option then :math:`0 \leq` ``var_PCA`` :math:`\leq 1` represents the fraction of preserved variance relative to the total variance in the data.
     - None
   * - ``alpha_reg``
     - float
     - Defines the penalty enforcement parameter :math:`\alpha` for the spectral regularisation term.
     - 1.0
   * - ``sumt_flag``
     - bool
     - Defines whether the sequential unconstrained minimisation technique is to be used during parameter estimation.
     - False
   * - ``sumt_parameters``
     - dict[str, float] | None
     - Defines a dictionary of parameters for the sequential unconstrained minimisation technique (SUMT) [3]. An example of this dictionary is given as {"alpha_init": 0.1, "alpha_end": 100, "alpha_multiplier": 10}.
     - None
   * - ``organise_by_kurt``
     - bool
     - Specifies whether source vectors :math:`\mathbf{w}_i` must be re-organised in descending order based on the kurtosis of :math:`z_i`.
     - False
   * - ``hessian_update_type``
     - str
     - Defines whether the actual Hessian is used or estimated via quasi-Newton methods [3]. The user can choose between four options: "actual", "SR1", "DFP", or "BFGS".
     - "actual"
   * - ``use_ls``
     - bool
     - Defines whether the step size :math:`\gamma_i^{(k)}` must be determined automatically or whether a user-defined value, set in the :math:`.\text{fit}(\cdot)` method call, must be used.
     - True
   * - ``second_order``
     - bool
     - Defines whether the optimisation algorithm is a first-order method or a second-order method. If the user chooses to set ``second_order = False``, then it is recommended that the default step size for gradient descent can be set in the :math:`.\text{fit}(\cdot)` method call.
     - True
   * - ``save_dir``
     - str | None
     - Defines whether visualisation of the properties through each the training iterations must be stored in some directory.
     - None
   * - ``verbose``
     - verbose
     -  Defines the verbosity mode for the model parameter estimation step.
     - False

Once an instance of the ``LinearModel`` class has been created, the ``.fit(.)`` method can be used to estimate the LVM parameters. The arguments for this method given below.

.. list-table:: The arguments for the ``.fit(.)`` call applied to an instance of the ``LinearModel`` class and their associated types, descriptions, and default values.
   :widths: 10 10 70 10
   :header-rows: 1

   * - Python API
     - Type
     - Description
     - Default value
   * - ``x_signal``
     - NumPy 1D array
     - Defines the single channel time-series signal that is to be used to estimate the LVM parameters.
     - Required argument
   * - ``n_iters``
     - int
     - The max number of iterations that are to be performed for each of the latent component vectors :math:`\mathbf{w}`.
     - 500
   * - ``learning_rate``
     - float
     - Defines a static value to the step size :math:`\gamma_i^{(k)}`. This is only used if enabled by the user, and will only come into effect if ``use_ls = False``.
     - 1.0
   * - ``tol``
     - float
     - Defines the tolerance of the latent component vector convergence. This is used to stop the iterations if the solution converges.
     - 1e-4
   * - ``use_tol``
     - bool
     - Defines a flag to specify if the convergence tolerance must be used. If ``use_tol = False``, the process will run for ``n_iters`` each time.
     - True
   * - ``Fs``
     - float
     - Defines the sampling frequency of the observed signal. Only used if the user wants to store visualisations of the solution vectors into the ``save_dir`` defined in the ``LinearModel`` instance.
     - 1.0

References
==========

[1.] Burden RL, Faires JD, Burden AM (2016) Numerical analysis, Tenth edition. Cengage Learning, Boston, MA

[2.] Artoni F, Delorme A, Makeig S (2018) `Applying dimension reduction to EEG data by principal component analysis reduces the quality of its subsequent independent component decomposition. <https://doi.org/10.1016/j.neuroimage.2018.03.016>`_ Neuroimage 175:176–187.

[3] Snyman JA, Wilke DN (2018) `Practical mathematical optimization <http://link.springer.com/10.1007/978-3-319-77586-9>`_, Second edition. Springer International Publishing, Cham
