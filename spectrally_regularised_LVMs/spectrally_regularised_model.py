# Copyright 2023-present Ryan Balshaw
"""
This script defines model parameter estimation process via the LinearModel class.
"""

import os
import time
from typing import TypeVar

import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scistats
from matplotlib import pyplot as plt
from tqdm import tqdm

from .cost_functions import CostClass
from .helper_methods import (
    BatchSampler,
    DataProcessor,
    DeflationOrthogonalisation,
    QuasiNewton,
)
from .spectral_regulariser import SpectralObjective


def initialise_W(n_sources: int, n_features: int, init_type: str):
    """
    A method that initialises the W matrix.

    Parameters
    ----------
    n_sources : int
        The number of source vectors to initialise.

    n_features : int
        The shape of the source vectors

    init_type : str
        The initialisation type for the vectors. Options are either
        'broadband' or 'random'. Broadband implies that vectors are
        dirac deltas, random implies that the vectors are randomly samples.

    Returns
    -------
    W : ndarray
        The initialised W matrix of shape (n_sources, n_features)
        that is normalised row-wise (ensures that each w vector is unit).
    """
    if init_type.lower() == "broadband":
        W = np.zeros((n_sources, n_features))

        # Set to one (impulse response)
        W[:, 0] = 1

    elif init_type.lower() == "random":
        W = np.random.randn(n_sources, n_features)

    else:
        print(f"Initialisation type ({init_type}) not understood.")
        raise SystemExit

    # Normalise rows
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    return W


def initialise_lambda(n_sources):
    """
    A method that initialises the lambda terms for lagrange expression.

    This is initialised to be a vector of ones.

    Returns
    -------
    Lambda: ndarray
        A vector of ones with shape (n_sources, 1)

    """
    Lambda = np.ones((n_sources, 1))

    return Lambda


cost_inst = TypeVar("cost_inst", bound=CostClass)


class LinearModel(object):
    """
    This class encapsulates the parameter estimation step of linear LVMs by defining
    a class that combines each of the different aspects of this package.

    Methods
    -------
    kurtosis(y)
        A method that calculates the kurtosis of a set of samples y.

    _function(param_vector, self_inst, W, X, idx)
        A static method that calculates the langrange expression.
        It is typeset so that it can interface with scipy.optimize.minimize
        methods.

    _gradient(param_vector, self_inst, W, X, idx)
        A static method that calculates the gradient of the langrange
        expression. It is typeset so that it can interface with
        scipy.optimize.minimize methods.

    _hessian(param_vector, self_inst, W, X, idx)
        A static method that calculates the Hessian of the langrange
        expression. It is typeset so that it can interface with scipy.optimize.minimize
        methods.

    line_search(delta, gradient, w, lambda_vector, W, X, idx)
        A method that performs a 1D line search on the delta vector to find a step
        size that satisfies the Armijo condition.

    lagrange_function(X, w, y, W, idx, lambda_vector)
        A method that calculates the lagrange expression.

    lagrange_gradient(X, w, y, W, idx, lambda_vector)
        A method that calculates the gradient of the lagrange expression.

    lagrange_hessian(X, w, y, W, idx, lambda_vector)
        A method that calculates the Hessian of the lagrange expression.

    parameter_update(self, X, w, y, W, idx, lambda_vector)
        A method that performs a parameter update based on the users optimisation
        properties.

    spectral_trainer(X, W, n_iters, learning_rate, tol, Lambda, Fs)
        This method estimates the model parameters for some X and W.

    update_params(w_current, lambda_current, delta_w, delta_lambda, W, idx)
        A method which calculates the update to w and lambda based off some global
        delta Phi vector, normalises w and performs GSO is requested.

    spectral_fit(X, W, n_iters = 1, learning_rate, tol, Lambda, Fs)
        A method that estimates the model parameters.

    fit(self, X, n_iters, learning_rate, tol, Fs)
        A method that uses spectral_fit, and allows users to use the sequential
        unconstrained minimisation technique (SUMT). This was done to make the
        API call similar to scikit-learn.

    transform(X)
        A method that transforms X to the latent domain via X @ W. If whitening is
        enabled the X represents an unwhitened matrix.

    inverse_transform(Z)
        A method that transforms samples from the latent domain to the data domain.
        If whitening is enabled then the recovered matrix represents the
        standardised data domain.

    compute_spectral_W(W)
        A static method that computes the spectral representations of the vectors
        in W.

    get_model_parameters()
        A method which return the solution parameters in a model_dict dictionary.

    set_model_parameters(model_dict, X)
        A method which sets the solution parameters based off the model_dict
        dictionary and X (X defines the pre-processing steps).
    """

    def __init__(
        self,
        n_sources: int,  # keep
        cost_instance: cost_inst,
        whiten: bool = True,  # keep
        init_type: str = "broadband",
        perform_gso: bool = True,
        batch_size: int | None = None,
        var_PCA: bool | None = None,
        alpha_reg: float = 1.0,
        sumt_flag: bool = False,
        sumt_parameters: dict[str, float] | None = None,
        organise_by_kurt: bool = False,
        hessian_update_type: str = "actual",
        use_ls: bool = True,
        use_hessian: bool = True,
        save_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        The initialisation of the linear_model class.

        Parameters
        ----------
        n_sources : int
            The number of latent sources that are to be estimated.

        cost_instance : instance which inherits from CostClass
            The objective function instance that the user chooses.

        whiten : bool (default = True)
            Specifies whether the data matrix X is to be whitened. By default, all X
            matrices are standardised.

        init_type : str (default = 'broadband')
            The initialisation strategy for W. Options are 'broadband' or 'random'.
            'broadband' sets each w vector to a dirac delta, while 'random' randomly
            initialises the vectors by sampling from an isotropic Gaussian
            distribution.

        perform_gso : bool (default = True)
            A flag to specify whether Gram-Schmidt Orthogonalisation is used during
            parameter estimation.

        batch_size : int | None (default = None)
            This variable allows users to perform stochastic optimisation if an
            integer value is entered. Default value implies all rows of X are used.

        var_PCA : float | None (default = None)
            This variable allows users to pre-process X by discarding any trailing
            eigenvalues. Default value of None implies that it is not used.

        alpha_reg : float (default = 1.0)
            Defines the penalty enforcement parameter applied to the spectral
            regularisation term.

        sumt_flag : bool (default = False)
            A flag that specifies whether the sequential unconstrained minimisation
            technique is to be used during parameter estimation.

        sumt_parameters : dict | None (default = None)
            The parameters for SUMT iteration. If default value is used, the
            dictionary is initialised to: {'alpha_init':0.1, 'alpha_end':10,
            'alpha_multiplier':100}. The expected keys for this dictionary are given
            in this docstring.

        organise_by_kurt : bool (default = False)
            A flag to control whether the vectors in W are to be organised by the
            kurtosis of the latent sources.

        hessian_update_type : str (default = 'actual')
            This specifies whether the actual Hessian is to be used during parameter
            estimation or if quasi-Newton strategies are to be employed. Options
            are: 'actual', 'SR1', 'DFP', 'BFGS'.

        use_ls : bool (default = True)
            A flag that specifies whether a linear line search is to be performed
            on the .

        use_hessian : bool (default = True)
            This flag specifies whether a second or first order optimisation method
            is used. If use_hessian = False then the method defaults to gradient
            descent.

        save_dir : str | None (default = None)
            Defines whether visualisation of the parameter estimation properties are
            to be stored in some directory. If a string is entered, this string is
            expected to be the directory in which the figures are to be saved.

        verbose : bool (default = False)
            Defines the verbosity mode for the parameter estimation step.
        """
        # Initialise instances
        self.n_sources = n_sources
        self.cost_instance = cost_instance  # ({'source_name':'exp', 'alpha':1})
        self.whiten = whiten
        self.var_PCA = var_PCA
        self.init_type = init_type.lower()
        self.organise_by_kurt = (
            organise_by_kurt  # A flag used to organise the ICA components
        )
        self.perform_gso = perform_gso
        self.batch_size = batch_size
        self.alpha_reg = alpha_reg
        self.sumt_flag = sumt_flag  # SUMT approach to alpha_reg
        self.sumt_parameters = sumt_parameters  # Parameters for SUMT updating
        self.hessian_update_type = (
            hessian_update_type.lower()
        )  # Type of jacobian update
        self.use_ls = use_ls
        self.use_hessian = use_hessian
        self.save_dir = save_dir
        self.verbose = verbose

        # Sequential unconstrained minimisation technique parameters
        if self.sumt_flag:
            if self.sumt_parameters is None:  # Initialise
                self.sumt_parameters = {
                    "alpha_init": 0.1,
                    "alpha_end": 100,
                    "alpha_multiplier": 10,
                }

            self.alpha_init = self.sumt_parameters["alpha_init"]
            self.alpha_end = self.sumt_parameters[
                "alpha_end"
            ]  # Overrides the default value
            self.alpha_multiplier = self.sumt_parameters["alpha_multiplier"]

            self.alpha_cnt = 0

        # Quasi-Newton solvers
        if self.hessian_update_type != "actual":
            self.quasi_newton_inst = QuasiNewton(
                self.hessian_update_type, use_inverse=True
            )
            print("Using a quasi-Newton iteration scheme.")

        if self.hessian_update_type != "actual" and self.use_hessian:
            print(
                "Selected quasi-Newton scheme but opted to "
                "not use hessian in update step."
                "\nDefaulting to quasi-Newton scheme."
            )

        if type(self.n_sources) != int:
            print("Please enter in a valid number of sources.")
            raise SystemExit

        if not self.whiten:
            if self.cost_instance.__class__.__name__ == "NegentropyCost":
                print("As negentropy loss is used, whitening is automatically applied.")
                # For future implementation, this could be adapted.
                self.whiten = True

            else:
                print("Non-whitened version is chosen.")

        # Initialise the processor instance  (could be in base class except for var_PCA)
        self.processor_inst = DataProcessor(self.whiten, self.var_PCA)

        if self.perform_gso:
            # Initialise the orthogonalisation instance (could be in base class)
            self.gs_inst = DeflationOrthogonalisation()

    def kurtosis(self, y):
        """

        Parameters
        ----------
        y : ndarray
            A vector or matrix of samples. If y is a vector, it is expected to be
            a column vector. If it is a matrix then each feature is given in
            a column.

        Returns
        -------
        kurtosis of the samples.
        """
        if self.whiten:  # y is zero-mean unit-variance
            if y.shape[1] == 1:
                return np.mean(y**4) - 3

            else:
                return np.mean(y**4, axis=0) - 3

        else:  # y is NOT zero-mean unit-variance
            if y.shape[1] == 1:
                return scistats.kurtosis(y, fisher=True)
            else:
                return scistats.kurtosis(y, axis=0, fisher=True)

    @staticmethod
    def _function(param_vector, self_inst, W, X, idx):
        """
        A static method that calculates the langrange expression. It is typeset so that
        it can interface with scipy.optimize.minimize methods.

        Parameters
        ----------
        param_vector : ndarray
            A vector that is given as [w.T, lambda].T

        self_inst : instance
            The instance of the LinearModel class.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        Returns
        -------
        The lagrangian expression evaluated at X and param_vector.
        """
        if len(param_vector.shape) == 1:  # (n,) shaped vector
            w, lambda_vector = param_vector[: W.shape[1]], param_vector[W.shape[1] :]

            w = w.reshape(-1, 1)
            lambda_vector = lambda_vector.reshape(-1, 1)

        else:
            w, lambda_vector = (
                param_vector[: W.shape[1], [0]],
                param_vector[W.shape[1] :, [0]],
            )

        # Project to the latent space
        y = np.dot(X, w)

        return self_inst.lagrange_function(X, w, y, W, idx, lambda_vector)

    @staticmethod
    def _gradient(param_vector, self_inst, W, X, idx):
        """
        A static method that calculates the gradient of the langrange expression.
        It is typeset so that it can interface with scipy.optimize.minimize methods.

        Parameters
        ----------
        param_vector : ndarray
            A vector that is given as [w.T, lambda].T

        self_inst : instance
            The instance of the LinearModel class.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        Returns
        -------
        The gradient of the lagrangian expression evaluated at X and param_vector.
        """
        # Used for scipy.optimize.line_search

        if len(param_vector.shape) == 1:  # (n,) shaped vector
            w, lambda_vector = param_vector[: W.shape[1]], param_vector[W.shape[1] :]

            w = w.reshape(-1, 1)
            lambda_vector = lambda_vector.reshape(-1, 1)

        else:
            w, lambda_vector = (
                param_vector[: W.shape[1], [0]],
                param_vector[W.shape[1] :, [0]],
            )

        # Project to the latent space
        y = np.dot(X, w)

        return self_inst.lagrange_gradient(X, w, y, W, idx, lambda_vector)[:, 0]

    @staticmethod
    def _hessian(param_vector, self_inst, W, X, idx):
        """
        A static method that calculates the Hessian of the langrange expression.
        It is typeset so that it can interface with scipy.optimize.minimize methods.

        Parameters
        ----------
        param_vector : ndarray
            A vector that is given as [w.T, lambda].T

        self_inst : instance
            The instance of the LinearModel class.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        Returns
        -------
        The Hessian of the lagrangian expression evaluated at X and param_vector.
        """
        # Used for scipy.optimize.line_search
        if len(param_vector.shape) == 1:  # (n,) shaped vector
            w, lambda_vector = param_vector[: W.shape[1]], param_vector[W.shape[1] :]

            w = w.reshape(-1, 1)
            lambda_vector = lambda_vector.reshape(-1, 1)

        else:
            w, lambda_vector = (
                param_vector[: W.shape[1], [0]],
                param_vector[W.shape[1] :, [0]],
            )

        # Project to the latent space
        y = np.dot(X, w)

        return self_inst.lagrange_hessian(X, w, y, W, idx, lambda_vector)

    def line_search(self, delta, gradient, w, lambda_vector, W, X, idx):
        """
        Performs a 1D line search on the delta vector to find a step size that satisfies
        the Armijo condition. Uses scipy.optimize.minimize routines.

        Parameters
        ----------
        delta : ndarray
            The parameter update vector.

        gradient : ndarray
            The gradient vector at the current iteration.

        w : ndarray
            The current w vector being optimised.

        lambda_vector : ndarray
            The current lambda_eq value being optimised.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        Returns
        -------
        alpha_val: float
            The step size that should be applied to delta.

        conv_flag: bool
            A flag that specifies whether the line search converged.
        """
        # Compute alpha parameter
        x0 = np.vstack((w.copy(), lambda_vector.copy()))[:, 0]
        c1 = 1e-4
        # c2 = 0.9
        # c3 = 0.9

        ls_dict = sciopt.linesearch.line_search_armijo(
            self._function,
            x0,
            delta[:, 0].copy(),
            gradient[:, 0].copy(),
            old_fval=None,
            args=(self, W, X, idx),
            c1=c1,
            alpha0=1,
        )

        # print("Finished.")
        # ls_dict = [None]

        # ls_dict = sciopt.line_search(self._function,
        #                               self._gradient,
        #                               x0,
        #                               delta[:, 0],
        #                               gradient[:, 0],
        #                               args=(self, W, X, idx),
        #                               c1=1e-4,
        #                               c2=0.9)

        if ls_dict[0] is not None:
            alpha_val = ls_dict[0]
            conv_flag = True

        # print(f"Line search converged, using to step size of {alpha_val}.")

        else:
            alpha_val = 1
            conv_flag = False
            print("Line search failed, defaulting to step size of 1.")
        # raise SystemExit

        return alpha_val, conv_flag

    def lagrange_function(self, X, w, y, W, idx, lambda_vector):
        """
        This method calculates the lagrangian expression.

        Parameters
        ----------
        X: ndarray
            The data matrix X.

        w : ndarray
            The current w vector being optimised.

        y : ndarray
            The transformed variable X @ w.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        lambda_vector : ndarray
            The current lambda_eq value being optimised.

        Returns
        -------
        The evaluation of the lagrangian expression.
        """
        # Compute ICA loss term
        objective_loss = self.cost_instance.cost(X, w, y)

        # Compute spectral loss term
        if idx > 0:
            idx_grid = np.ix_(np.arange(0, idx, 1), np.arange(W.shape[1]))

            spectral_loss = self.alpha_reg * np.sum(
                self.spectral_obj.spectral_loss(w, W[idx_grid]), axis=0
            )
        else:
            spectral_loss = 0

        # Compute the optimisation constraint term
        h_w = w.T @ w - 1
        constraint = lambda_vector @ h_w

        return objective_loss + spectral_loss + constraint[0, 0]

    def lagrange_gradient(self, X, w, y, W, idx, lambda_vector):
        """
        This method calculates the gradient of the lagrangian expression.
        Parameters
        ----------
        X: ndarray
            The data matrix X.

        w : ndarray
            The current w vector being optimised.

        y : ndarray
            The transformed variable X @ w.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        lambda_vector : ndarray
            The current lambda_eq value being optimised.

        Returns
        -------
        The evaluation of the gradient of the lagrangian expression.
        """
        # t0 = time.time()
        grad_cost = self.cost_instance.cost_gradient(X, w, y)
        # t1 = time.time()
        # print(f"Computing the negentropy derivative
        # took {np.round(t1 - t0, 5)} seconds")

        # Compute the first half
        h_w = w.T @ w - 1

        dh_dw = 2 * w

        # Compute the Lagrangian function gradient vector
        grad_lagrange = grad_cost + dh_dw @ lambda_vector

        if idx > 0:
            # t0 = time.time()
            idx_grid = np.ix_(np.arange(0, idx, 1), np.arange(W.shape[1]))

            grad_lagrange += self.alpha_reg * np.sum(
                self.spectral_obj.spectral_derivative(w, W[idx_grid]),
                axis=1,
                keepdims=True,
            )
        # t1 = time.time()
        # print(f"Computing the spectral derivative took
        # {np.round(t1 - t0, 5)} seconds")

        # Stack the two derivatives
        grad_vector = np.vstack((grad_lagrange, h_w))

        return grad_vector

    def lagrange_hessian(self, X, w, y, W, idx, lambda_vector):
        """
        This method calculates the Hessian of the lagrangian expression.
        Parameters
        ----------
        X: ndarray
            The data matrix X.

        w : ndarray
            The current w vector being optimised.

        y : ndarray
            The transformed variable X @ w.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        lambda_vector : ndarray
            The current lambda_eq value being optimised.

        Returns
        -------
        The evaluation of the Hessian of the lagrangian expression.
        """

        jac_cost = self.cost_instance.cost_hessian(X, w, y)

        # Compute the jacobian of the Lagrangian gradient vector
        jac_lagrange = jac_cost + (2 * lambda_vector[0, 0] * np.eye(w.shape[0]))
        dh_dw = [2 * w]

        if idx > 0:  # Add in the spectral regularisation
            jac_lagrange += self.alpha_reg * self._hessian_regulariser

        J2 = np.hstack(dh_dw).T

        j_top = np.hstack((jac_lagrange, J2.T))
        j_bot = np.hstack((J2, np.zeros(2 * [lambda_vector.shape[0]])))
        jacobian = np.vstack((j_top, j_bot))

        return jacobian

    def parameter_update(self, X, w, y, W, idx, lambda_vector):
        """
        The method get an updated estimate of the parameters. Combines all the
        user choices into one simple step. It accounts for standard gradient descent,
        stochastic gradient descent, second order methods, and quasi-second order
        methods.

        Parameters
        ----------
        X: ndarray
            The data matrix X.

        w : ndarray
            The current w vector being optimised.

        y : ndarray
            The transformed variable X @ w.

        W : ndarray
            The matrix of w vectors stored in the rows of W.

        X : ndarray
            The data matrix.

        idx : int
            The current iteration index for the parameters.

        lambda_vector : ndarray
            The current lambda_eq value being optimised.

        Returns
        -------
        delta_w : ndarray
            The update that should be applied to w.

        delta_lambda: ndarray
            The update that should be applied to lambda.

        gradient : ndarray
            The gradient evaluation at the current iteration index.
        """
        # Define flag that allows for solver re-initialisation
        self._reinit_flag = False

        # Calculate the gradient
        gradient = self.lagrange_gradient(X, w, y, W, idx, lambda_vector)

        if self.use_hessian:
            if self.hessian_update_type == "actual":
                jacobian = self.lagrange_hessian(X, w, y, W, idx, lambda_vector)

                # Compute the update
                delta = -1 * np.linalg.solve(jacobian, gradient)

            else:
                # Compute the delta term using quasi-Newton approach
                delta = self.quasi_newton_inst.compute_update(gradient)

        else:
            jacobian = np.eye(gradient.shape[0])

            # Compute the update
            delta = -1 * np.linalg.solve(
                jacobian, gradient
            )  # Should just be negative gradient

        # Compute alpha parameter
        if self.use_ls:
            alpha_multiplier, conv_flag = self.line_search(
                delta, gradient, w, lambda_vector, W, X, idx
            )

        else:
            alpha_multiplier = self._training_learning_rate
            conv_flag = True

        # Scale delta by alpha
        delta = alpha_multiplier * delta

        # Split into W and Lambda updates
        delta_w, delta_lambda = delta[: w.shape[0], [0]], delta[w.shape[0] :, [0]]

        # Compute quasi-Newton updates, if required
        if self.hessian_update_type != "actual" and self.use_hessian:
            if conv_flag:  # Compute quasi-Newton updates.
                # Compute the new parameters
                w_new, lambda_new = self.update_params(
                    w.copy(), lambda_vector.copy(), delta_w, delta_lambda, W, idx
                )

                # Compute gradient_diff_k
                gradient_next = self.lagrange_gradient(X, w_new, y, W, idx, lambda_new)
                gradient_diff_k = gradient_next - gradient

                # Update the Hessian approximation
                self.quasi_newton_inst.update_jacobian(delta, gradient_diff_k)

            else:
                if self.verbose:
                    print("\n Line search did not converge.")

                delta_w *= 0.0  # Don't perform any update
                delta_lambda *= 0.0  # Don't perform any update

                self._reinit_flag = True  # Set to true to allow solver to continue

                self.quasi_newton_inst.initialise_jacobian(
                    delta.shape[0]
                )  # Re-initialise the Jacobian

        return delta_w, delta_lambda, gradient

    def spectral_trainer(self, X, W, n_iters, learning_rate, tol, Lambda, Fs):
        """
        This method estimates the model parameters for some X and W.

        Parameters
        ----------
        X : ndarray
            The data matrix X.

        W : ndarray
                The source vector matrix W.

        n_iters : int
                The max number of iterations that are to be performed for each source.

        learning_rate : float
                A learning rate. This is only used if required by the user, and will
                only appear if use_ls is not activated.

        tol : float
                The tolerance on the convergence error, error =| w_new^T @ w_prev - 1|.
                Used to stop the solver if it converges.

        Lambda : ndarray
                A vector of lambda parameters for the Lagrange expressions.

        Fs : float
                The sampling frequency of the observed signal. Only used if the user
                wants to store visualisations of the solution vectors.

        Returns
        -------
        W_ : ndarray
                The estimated source vectors.

        Lambda_: ndarray
                The estimated lambda values for the lagrange expressions.
        """
        # Initialise training metrics
        self.kurtosis_ = []
        self.variance_ = []
        self.grad_norm_ = []
        self.cost_ = []
        self.lagrange_cost_ = []
        self.w_similarity_ = []
        self.spectral_loss_ = []

        # Done to ensure that we do not update W (numpy memory pointing effect)
        W_ = W.copy()
        Lambda_ = Lambda.copy()

        # Initialise the stored hessians
        self._training_learning_rate = learning_rate
        self.gradient_store = None

        # Setup lambda and storage vectors.
        for idx in range(0, W_.shape[0], 1):
            w_new = W_[[idx], :].T.copy()

            lambda_new = Lambda_[[idx], :].T.copy()  # Should just be a 1 x 1 vector

            grad_norm = np.zeros(n_iters)
            kurtosis = np.zeros(n_iters)
            variance = np.zeros(n_iters)
            obj_cost = np.zeros(n_iters)
            lagrange_loss = np.zeros(n_iters)
            w_similarity = np.zeros(n_iters)
            spectral_loss = np.zeros(n_iters)

            if idx > 0:
                # Make grid to extract from W_
                ix_grid = np.ix_(np.arange(0, idx, 1), np.arange(W_.shape[1]))

                # Compute the Hessian
                self._hessian_regulariser = np.sum(
                    self.spectral_obj.spectral_hessian(w_new, W_[ix_grid].copy()),
                    axis=0,
                )

            # Update
            if self.hessian_update_type != "actual":
                self.quasi_newton_inst.initialise_jacobian(W.shape[1] + 1)

            self.cnt_iter = 0
            error = np.inf

            if self.verbose:
                pbar = tqdm(total=n_iters)

            while self.cnt_iter < n_iters and error >= tol:
                # t0 = time.time()

                if self.batch_size is None:
                    Y = np.dot(X, w_new)

                    delta_w, delta_lambda, gradient = self.parameter_update(
                        X,
                        w_new,
                        Y,
                        W_,
                        idx,
                        lambda_new,  # X, w, y, W, idx, lambda_vector
                    )

                else:
                    Xi = next(
                        self._data_sampler
                    )  # Perhaps this should be reinitialised within this method?
                    Y = np.dot(Xi, w_new)
                    delta_w, delta_lambda, gradient = self.parameter_update(
                        Xi, w_new, Y, W_, idx, lambda_new
                    )

                w_prev = w_new.copy()

                w_new, lambda_new = self.update_params(
                    w_new, lambda_new, delta_w, delta_lambda, W_, idx
                )

                # Generate stored metrics
                y_new = np.dot(X, w_new)
                grad_norm[self.cnt_iter] = np.linalg.norm(gradient)
                kurtosis[self.cnt_iter] = self.kurtosis(y_new)
                variance[self.cnt_iter] = np.std(y_new)
                obj_cost[self.cnt_iter] = self.cost_instance.cost(X, w_new, y_new)
                lagrange_loss[self.cnt_iter] = self.lagrange_function(
                    X, w_new, y_new, W_, idx, lambda_new
                )
                w_similarity[self.cnt_iter] = np.abs(w_new.T @ w_prev - 1)[0, 0]

                # Compute spectral loss term
                if idx > 0:
                    idx_grid = np.ix_(np.arange(0, idx, 1), np.arange(W_.shape[1]))
                    spectral_loss[self.cnt_iter] = self.alpha_reg * np.sum(
                        self.spectral_obj.spectral_loss(w_new, W_[idx_grid]), axis=0
                    )

                else:
                    spectral_loss[self.cnt_iter] = 0

                # t1 = time.time()
                # print(f"Calculate metrics: {t1 - t0} seconds")
                # t0 = time.time()

                if self.cnt_iter > 5:
                    error = w_similarity[self.cnt_iter]

                # Update the iterator
                if self.verbose:
                    pbar.update(1)
                    pbar.set_description(
                        f"Component {idx + 1} - " f"Error: {np.round(error, 6)}"
                    )

                # Update counter
                self.cnt_iter += 1

                if self._reinit_flag:
                    error = (
                        np.inf
                    )  # Stops solver from terminating on quasi-Newton re-initialisation

            if self.save_dir is not None:
                fig, ax = plt.subplots(4, 2, figsize=(8, 12))
                ax = ax.flatten()

                # Create the FFT properties
                n = w_new.shape[0]
                freq = np.fft.fftfreq(n, 1 / Fs)[: n // 2]
                vals = 2 / n * np.abs(np.fft.fft(w_new[:, 0]))[: n // 2]

                # Plot
                ax[0].set_title("Spectral content")
                ax[0].plot(freq, vals, color="#003f5c")

                ax[1].set_title(r"$\Delta x$ norm")
                ax[1].semilogy(grad_norm[: self.cnt_iter], color="#444e86")

                ax[2].set_title("Latent variance")
                ax[2].plot(variance[: self.cnt_iter], color="#955196")

                ax[3].set_title("Latent kurtosis")
                ax[3].plot(kurtosis[: self.cnt_iter], color="#955196")

                ax[4].set_title("Original loss")
                ax[4].plot(obj_cost[: self.cnt_iter], color="#dd5182")

                ax[5].set_title("Lagrange function loss")
                ax[5].plot(lagrange_loss[: self.cnt_iter], color="#ff6e54")

                ax[6].set_title("Update similarity")
                ax[6].plot(w_similarity[: self.cnt_iter], color="#ffa600")

                ax[7].set_title("Spectral regularisation")
                ax[7].plot(spectral_loss[: self.cnt_iter], color="b")

                for axs in ax:
                    axs.grid()

                for axs in ax[1:]:
                    axs.set_xlabel("Iteration index")

                ax[0].set_xlabel("Index")

                fig.tight_layout()

                if self.sumt_flag:
                    save_name = f"component_{idx}_alpha={self.alpha_cnt}.png"
                else:
                    save_name = f"component_{idx}.png"

                plt.savefig(os.path.join(self.save_dir, save_name))

                plt.close("all")

            # Close the iterator
            if self.verbose:
                pbar.close()

            # Store W and lambda
            W_[idx, :] = w_new[:, 0]

            Lambda_[idx, :] = lambda_new.copy()  # Should just be a 1 x 1 vector

            # t0 = time.time()
            self.kurtosis_.append(kurtosis[: self.cnt_iter])
            self.variance_.append(variance[: self.cnt_iter])
            self.grad_norm_.append(grad_norm[: self.cnt_iter])
            self.cost_.append(obj_cost[: self.cnt_iter])
            self.lagrange_cost_.append(lagrange_loss[: self.cnt_iter])
            self.w_similarity_.append(w_similarity[: self.cnt_iter])
            self.spectral_loss_.append(spectral_loss[: self.cnt_iter])

        # Return the updated matrices
        return W_, Lambda_

    def update_params(self, w_current, lambda_current, delta_w, delta_lambda, W, idx):
        """
        A method that computes the update to the w and lambda parameters, performs GSO
        if required by the user and ensures that w is a unit vector.

        Parameters
        ----------
        w_current : ndarray
                The current source vector.

        lambda_current : ndarray
                The current lambda value for the Lagrangian expression.

        delta_w : ndarray
                The update to be applied to the source vector.

        delta_lambda : ndarray
                The update to be applied to the lambda value.

        W : ndarray
                The W matrix of source vectors.

        idx : int
                The index of the source vector in W that is currently being solved for.

        Returns
        -------
        w_new : ndarray
                The updated w vector.

        lambda_new :
                The updated lambda value.
        """
        # Update W_new
        w_new = w_current + delta_w

        # Update lambda
        lambda_new = lambda_current + delta_lambda

        # Normalise
        w_new /= np.linalg.norm(w_new)

        # Perform GSO if whitening flag is enabled.
        if idx > 0 and self.perform_gso:
            # t0 = time.time()
            w_new = self.gs_inst.gram_schmidt_orthogonalisation(w_new, W, idx)
        # t1 = time.time()
        # print(f"Perform GSO: {t1 - t0} seconds")

        return w_new, lambda_new

    def spectral_fit(
        self,
        X,
        W,
        n_iters=1,
        learning_rate=0.1,
        tol=1e-3,
        Lambda=None,
        Fs: float | int = 25e3,
    ):
        """
        This method estimates the model parameters for some given X, W, and Lambda.

        Parameters
        ----------
        X : ndarray
            The data matrix X.

        W : ndarray
                The source vector matrix W.

        n_iters : int
                The max number of iterations that are to be performed for each source.

        learning_rate : float
                A learning rate. This is only used if required by the user, and will
                only appear if use_ls is not activated.

        tol : float
                The tolerance on the convergence error, error =| w_new^T @ w_prev - 1|.
                Used to stop the solver if it converges.

        Lambda : ndarray
                A vector of lambda parameters for the Lagrange expressions.

        Fs : float
                The sampling frequency of the observed signal. Only used if the user
                wants to store visualisations of the solution vectors.

        Returns
        -------
        W_update : ndarray
                The estimated W matrix.

        Lambda_update : ndarray
                The estimated Lambda values.

        """
        # Call the spectral trainer
        W_update, Lambda_update = self.spectral_trainer(
            X, W, n_iters, learning_rate, tol, Lambda, Fs
        )

        # Calculate the excess kurtosis
        Y = np.dot(X, W_update.T)
        kurt = np.mean(Y**4, axis=0) - 3  # Excess kurtosis

        if self.organise_by_kurt:
            pos_idx = np.argsort(kurt)[::-1]
            kurt = kurt[pos_idx]

            W_update = W_update[pos_idx, :]
            if self.verbose:
                print("\nOrganised W by source excess kurtosis.")

        # Visualise the sICA results (in spectral_fit method
        # because of SUMT iteration)
        if self.save_dir is not None:
            fig, ax = plt.subplots(6, 5, figsize=(12, 14))
            fig.suptitle(
                f"First 30 components (regularisation parameter = {self.alpha_reg})"
            )
            ax = ax.flatten()
            n = X.shape[1]

            freq = np.fft.fftfreq(n, 1 / Fs)[: n // 2]
            for cnt, i in enumerate(range(min(self.n_sources, 30))):
                ax[cnt].plot(
                    freq,
                    2 / n * np.abs(np.fft.fft(W_update[i, :]))[: n // 2],
                    color="b",
                )
                ax[cnt].grid()
                ax[cnt].set_xlabel("Frequency (Hz)")
                ax[cnt].set_title(f"Excess kurtosis: {np.round(kurt[i], 3)}")
            fig.tight_layout()

            if self.sumt_flag:
                save_name = f"spectral_W_alpha={self.alpha_cnt}.png"

            else:
                save_name = "spectral_W_final.png"
            plt.savefig(os.path.join(self.save_dir, save_name))
            plt.close("all")

        return W_update, Lambda_update

    def fit(
        self,
        X,
        n_iters=100,
        learning_rate=1,
        tol=1e-4,
        Fs: int | float = 25e3,
    ):
        """
        This method follows the scikit-learn API call and estimates the model parameters
        based off the users initialisation choices.

        Parameters
        ----------
        X : ndarray
            The data matrix X.

        n_iters : int
                        The max number of iterations that are to be performed for each
                        source.

        learning_rate : float
                        A learning rate. This is only used if required by the user, and
                        will only appear if use_ls is not activated.

        tol : float
                        The tolerance on the convergence error,
                        error =| w_new^T @ w_prev - 1|. Used to stop the solver
                        if it converges.

        Fs : float
                        The sampling frequency of the observed signal. Only used if
                        the user wants to store visualisations of the solution vectors.

        Returns
        -------
        self : instance
                        This method returns self so that it can be chained onto the
                        initialisation of the class via
                        model_inst = LinearModel(...).fit(...).

        """
        # Initialise pre-processing
        self.processor_inst.initialise_preprocessing(X)

        # Pre-process the data
        X_preprocess = self.processor_inst.preprocess_data(X)

        # Get number of sources
        N, m = X_preprocess.shape

        # Initialise W
        W = initialise_W(self.n_sources, m, self.init_type)

        # Initialise Lambda
        Lambda = initialise_lambda(self.n_sources)

        # Initialise spectral objective instance
        self.spectral_obj = SpectralObjective(
            m, save_hessian_flag=False, inv_hessian_flag=False, verbose=False
        )

        # Initialise the batch sampler
        if self.batch_size is not None:
            batch_sampler_inst = BatchSampler(self.batch_size, include_end=True)
            self._data_sampler = iter(batch_sampler_inst(X_preprocess, iter_idx=0))

        # Train based on the users choice (SUMT or standard)
        if self.sumt_flag:
            param_iters = []
            W_iters = [W.copy()]
            lambda_iters = [Lambda.copy()]
            spectral_W_iters = [self.compute_spectral_W(W)]
            solution_error = []

            # Initialise the penalty parameter term
            self.alpha_reg = self.alpha_init

            while self.alpha_reg <= self.alpha_end:
                W_update, Lambda_update = self.spectral_fit(
                    X_preprocess,
                    W_iters[-1],
                    n_iters,
                    learning_rate,
                    tol,
                    lambda_iters[-1],
                    Fs,
                )

                # Store solution and parameters
                param_iters.append(
                    (
                        self.kurtosis_,
                        self.variance_,
                        self.grad_norm_,
                        self.cost_,
                        self.lagrange_cost_,
                        self.w_similarity_,
                        self.spectral_loss_,
                    )
                )

                # Store the iteration parameters
                W_iters.append(W_update.copy())
                lambda_iters.append(Lambda_update.copy())

                # Get spectral representation of W
                spectral_W = self.compute_spectral_W(W_update)
                spectral_W_iters.append(spectral_W)

                solution_error.append(
                    np.mean(np.linalg.norm(W_iters[-1] - W_iters[-2], axis=1))
                )

                # Display the solution error
                if self.verbose:
                    print("----" * 10)
                    print(
                        f"\nFor alpha = {self.alpha_reg}, "
                        f"error = {np.round(solution_error[-1], 4)}.\n"
                    )
                    print("----" * 10, "\n")

                    time.sleep(1)

                # Update counter
                self.alpha_cnt += 1

                # Update alpha
                self.alpha_reg = self.alpha_reg * self.alpha_multiplier

            # Store actual and value over iterations.
            pos_min = np.argmin(solution_error)
            self.solution_error = solution_error
            (
                self.kurtosis_,
                self.variance_,
                self.grad_norm_,
                self.cost_,
                self.lagrange_cost_,
                self.w_similarity_,
                self.spectral_loss_,
            ) = param_iters[pos_min]

            # Check pos_min

            self.Lambda = lambda_iters[pos_min + 1]
            self.W = W_iters[
                pos_min + 1
            ]  # un-adjust pos_min as W_iters has an initial point.

            self.spectral_W = spectral_W_iters[pos_min + 1]

            # Store for looking at later.
            self.param_iters = param_iters
            self.lambda_iters = lambda_iters
            self.W_iters = W_iters
            self.spectral_W_iters = spectral_W_iters

        else:
            W_update, Lambda_update = self.spectral_fit(
                X_preprocess,
                W,
                n_iters,
                learning_rate,
                tol,
                Lambda,
                Fs,
            )

            # Store the parameters
            self.W = W_update
            self.Lambda = Lambda_update
            self.spectral_W = self.compute_spectral_W(self.W)

        # Compute the excess kurtosis of the model
        Y = np.dot(X_preprocess, self.W.T)

        self.excess_kurtosis_ = np.mean(Y**4, axis=0) - 3  # Excess kurtosis

        return self

    def transform(self, X):
        """
        This method transforms a data matrix X to the latent space.

        Parameters
        ----------
        X : ndarray
            The data matrix X.

        Returns
        -------
        Z : ndarray
            The projection of X to the latent space.
        """

        Z = self.processor_inst.preprocess_data(X) @ self.W.T

        return Z

    def inverse_transform(self, Z, full_inverse: bool = False):
        """
        This method transforms a latent matrix X to the latent space.

        Parameters
        ----------
        Z : ndarray
            The latent matrix Z.

        full_inverse : bool
            A flag to specify whether the recovered data matrix X must be
            returned in standardised form or in the original, un-standardised
            domain.

        Returns
        -------
        X_recon : ndarray
            The reconstruction of X from the latent matrix Z.
        """
        Z_ = Z.copy()

        # Transform back to the training feature space
        X_recon = np.dot(Z_, self.W)

        # Un-process the data (still zero-mean, unit-variance)
        X_recon = self.processor_inst.unprocess_data(X_recon)

        if full_inverse:
            X_recon = (X_recon * self.processor_inst.std_) + self.processor_inst.mean_

        return X_recon

    @staticmethod
    def compute_spectral_W(W):
        """
        This method computes the spectral representation of the vectors in the W
        matrix.

        Parameters
        ----------
        W : ndarray
                The source vector matrix W.

        Returns
        -------
        spectral_W : ndarray
                A matrix that contains the spectral magnitude information of the
                sources.
        """
        # Get the spectral content of the filters
        r, c = W.shape

        spectral_W = np.zeros((r, c // 2))  # Only use one half as filter is real.
        # spectral_W_PCA = np.zeros((r, c//2))

        for i in range(0, r, 1):
            val = (
                2 / c * np.abs(np.fft.fft(W[i, :])[: c // 2])
            )  # Use this if you want to use ICA
            # val = 2/n * np.abs(
            #     np.fft.fft(ICA_model.Vh[i, :])[:n//2]
            # ) # Use this if you want to use PCA
            spectral_W[i, :] = val

        return spectral_W

    def get_model_parameters(self):
        """
        This method gets all the important model parameters and .

        Returns
        -------
        dict_params : dict
                A dictionary which stores all the solution information.
        """
        dict_params = {
            "W": self.W,
            "spectral_W": self.spectral_W,
            "kurtosis_": self.kurtosis_,
            "variance_": self.variance_,
            "grad_norm_": self.grad_norm_,
            "cost_": self.cost_,
            "lagrange_cost_": self.lagrange_cost_,
            "w_similarity_": self.w_similarity_,
            "spectral_loss_": self.spectral_loss_,
            "excess_kurtosis_": self.excess_kurtosis_,
        }

        if self.sumt_flag:
            dict_params["solution_error"] = self.solution_error
            dict_params["W_iters"] = self.W_iters
            dict_params["spectral_W_iters"] = self.spectral_W_iters
            dict_params["param_iters"] = self.param_iters
            dict_params["lambda_iters"] = self.lambda_iters
            dict_params["alpha_values"] = 10.0 ** (
                np.arange(
                    np.log10(self.sumt_parameters["alpha_init"]),
                    np.log10(self.sumt_parameters["alpha_end"]) + 1,
                    1,
                )
            )

        return dict_params

    def set_model_parameters(self, X, dict_params: dict):
        """
        This method takes the X matrix and a parameter dictionary, initialises the
        pre-processing components and then creates the necessary class attributes
        from the dictionary.

        Parameters
        ----------
        X : ndarray
            The data matrix X.

        dict_params : dict
                The parameter dictionary that is returned by the .get_model_parameters()
                method.

        Returns
        -------
        self : instance
                This method returns self so that it can be chained onto the
                initialisation of the class via
                model_inst = LinearModel(...).set_model_parameters(...).
        """

        # Initialise pre-processing
        self.processor_inst.initialise_preprocessing(X)

        # Store dictionary
        for k, v in dict_params.items():
            setattr(self, k, v)

        return self
