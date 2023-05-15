# Copyright 2023-present Ryan Balshaw
"""
A complete implementation of the scICA method.
"""

import os
import time
from typing import TypeVar

import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scistats
from matplotlib import pyplot as plt
from tqdm import tqdm

from .cost_functions import costClass
from .helper_methods import (
    batch_sampler,
    data_processor,
    deflation_orthogonalisation,
    quasi_Newton,
)
from .spectral_constraint import spectral_objective


def initialise_W(n_sources, m, init_type):
    """initialise_W docstring"""
    if init_type == "broadband":
        W = np.zeros((n_sources, m))

        # Set to one (impulse response)
        W[:, 0] = 1

    elif init_type == "random":
        W = np.random.randn(n_sources, m)

    else:
        print(f"Initialisation type ({init_type}) not understood.")
        raise SystemExit

    # Normalise rows
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    return W


def initialise_lambda(n_sources):
    """initialise_lambda docstring"""
    Lambda = np.ones((n_sources, 1))  # np.zeros((n_sources, 1)) #

    return Lambda


class linear_model(object):
    """Linear model docstring"""

    def __init__(
        self,
        n_sources: int,  # keep
        cost_instance: TypeVar("cost_inst", bound=costClass),  # keep
        whiten: bool = True,  # keep
        init_type: str = "broadband",  # broadband or random - keep
        organise_by_kurt: bool = False,  # keep
        perform_gso: bool = True,  # keep
        batch_size: int | None = None,  # keep
        var_PCA: bool | None = None,  # keep
        alpha_reg: float = 1.0,  # keep
        sumt_flag: bool = False,  # keep
        sumt_parameters: dict[str, float] = {  # keep
            "alpha_init": 0.1,
            "alpha_end": 10,
            "alpha_multiplier": 10,
        },
        hessian_update_type: str = "full",  # full, SR1, DFP, BFGS
        use_ls: bool = True,  # keep
        use_hessian: bool = True,  # Controls whether SGD or Newton
        save_dir: bool | None = None,  # Not sure
        verbose: bool = False,  # keep
    ):
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

        # Initialise the processor instance  (could be in base class except for var_PCA)
        self.processor_inst = data_processor(self.whiten, self.var_PCA)

        if self.perform_gso:
            # Initialise the orthogonalisation instance (could be in base class)
            self.gs_inst = deflation_orthogonalisation()

        # Sequential unconstrained minimisation technique parameters
        if self.sumt_flag:
            # self.eps_1 = self.sumt_parameters["eps_1"]
            # self.eps_2 = 1e-3 # Not applicable here
            self.alpha_init = self.sumt_parameters["alpha_init"]  # 10**-6
            self.alpha_end = self.sumt_parameters[
                "alpha_end"
            ]  # Overrides the default value
            self.alpha_multiplier = self.sumt_parameters["alpha_multiplier"]

            self.alpha_cnt = 0

        # Quasi-Newton solvers
        if self.hessian_update_type != "full":
            self.quasi_newton_inst = quasi_Newton(
                self.hessian_update_type, use_inverse=True
            )
            print("Using a quasi-Newton iteration scheme.")

        if self.hessian_update_type != "full" and self.use_hessian:
            print(
                "Selected quasi-Newton scheme but opted to "
                "not use hessian in update step."
                "\nDefaulting to quasi-Newton scheme."
            )

        if type(self.n_sources) != int:
            print("Please enter in a valid number of sources.")
            raise SystemExit

        if not self.whiten:
            print("Non-whitened version is chosen.")

    def kurtosis(self, y):  # Important to FastICA
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
    def _function(
        param_vector, self_inst, W, X, idx
    ):  # Used for scipy.optimize.line_search
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

    def line_search(
        self, delta, gradient, w, lambda_vector, W, X, idx, visualise=False
    ):
        ## Compute alpha parameter
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

        # Compute the constraint term
        h_w = w.T @ w - 1
        constraint = lambda_vector @ (h_w)

        return objective_loss + spectral_loss + constraint[0, 0]

    def lagrange_gradient(self, X, w, y, W, idx, lambda_vector):
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
        jac_cost = self.cost_instance.cost_hessian(X, w, y)

        # Compute the jacobian of the Lagrangian gradient vector
        jac_lagrange = jac_cost + (2 * lambda_vector[0, 0] * np.eye(w.shape[0]))
        dh_dw = [2 * w]

        if idx > 0:  # Add in the spectral constraint
            jac_lagrange += self.alpha_reg * self._hessian_constraint

        J2 = np.hstack(dh_dw).T

        j_top = np.hstack((jac_lagrange, J2.T))
        j_bot = np.hstack((J2, np.zeros(2 * [lambda_vector.shape[0]])))
        jacobian = np.vstack((j_top, j_bot))

        return jacobian

    def Newton_update(self, X, w, y, W, idx, lambda_vector):
        if self.gradient_store is None:
            # t0 = time.time()
            gradient = self.lagrange_gradient(X, w, y, W, idx, lambda_vector)
            # t1 = time.time()
            # print(f"Calculate derivative: {t1 - t0} seconds")

        else:
            gradient = self.gradient_store  # Use the new stored gradient

        if self.hessian_update_type == "full":
            # t0 = time.time()

            if self.use_hessian:
                jacobian = self.lagrange_hessian(X, w, y, W, idx, lambda_vector)

            else:
                jacobian = np.eye(gradient.shape[0])

            # t1 = time.time()
            # print(f"Calculate Jacobian: {t1 - t0} seconds")

            # t0 = time.time()
            delta = -1 * np.linalg.solve(jacobian, gradient)
            # t1 = time.time()
            # print(f"Solve for delta: {t1 - t0} seconds")

            ## Compute alpha parameter
            if self.use_ls:
                alpha_multiplier, conv_flag = self.line_search(
                    delta, gradient, w, lambda_vector, W, X, idx, visualise=False
                )
                # alpha_multiplier = 1
                # bs_opt = self.backtracking(delta,
                #                            gradient,
                #                            w,
                #                            lambda_vector,
                #                            W,
                #                            X,
                #                            idx,
                #                            max_iter=100)
                #
                # if bs_opt[0] is not None:
                #     alpha_multiplier = bs_opt[0]
                #     print(alpha_multiplier)

            else:
                alpha_multiplier = (
                    self._training_learning_rate
                )  # Use the prescribed learning rate.

            # Scale delta by alpha
            delta = alpha_multiplier * delta

            # if not conv_flag:
            #     print("\n Line search did not converge.")

            ## Split into W and Lambda
            delta_w, delta_lambda = delta[: w.shape[0], [0]], delta[w.shape[0] :, [0]]

        else:
            # t0 = time.time()

            ## Compute the delta term
            delta = self.quasi_newton_inst.compute_update(gradient)

            ## Compute alpha parameter
            alpha, conv_flag = self.line_search(
                delta, gradient, w, lambda_vector, W, X, idx
            )
            # alpha = 1

            if conv_flag:
                # Scale delta by alpha
                delta = alpha * delta

            else:
                if self.verbose:
                    print("\n Line search did not converge.")

                delta *= 0.0  # Don't perform any update
                self.quasi_newton_inst.initialise_jacobian(
                    delta.shape[0]
                )  # Re-initialise the Jacobian

            ## Split into W and Lambda
            delta_w, delta_lambda = delta[: w.shape[0], [0]], delta[w.shape[0] :, [0]]

            ## Compute the new parameters
            w_new, lambda_new = self.update_params(
                w.copy(), lambda_vector.copy(), delta_w, delta_lambda, W, idx
            )

            ## Compute gradient_diff_k
            gradient_next = self.lagrange_gradient(X, w_new, y, W, idx, lambda_new)
            gradient_diff_k = gradient_next - gradient
            self.gradient_store = gradient_next

            # t1 = time.time()
            # print(f"Calculate Jacobian: {t1 - t0} seconds")

            ## Update the Hessian approximation
            # t0 = time.time()
            self.quasi_newton_inst.update_jacobian(delta, gradient_diff_k)

            # t1 = time.time()
        # if idx > 1:
        #     print(f"Calculate delta: {t1 - t0} seconds")
        #
        # if t1 - t0 > 3:
        #     print("Hit a break, check out why it is taking so long.")

        return delta_w, delta_lambda, gradient

    def spectral_trainer(self, X, W, n_iters, learning_rate, tol, Lambda, Fs):
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
                self._hessian_constraint = np.sum(
                    self.spectral_obj.spectral_hessian(w_new, W_[ix_grid].copy()),
                    axis=0,
                )

            # Update
            if self.hessian_update_type != "full":
                self.quasi_newton_inst.initialise_jacobian(self.n_sources + 1)

            self.cnt_iter = 0
            error = np.inf

            if self.verbose:
                pbar = tqdm(total=n_iters)

            while self.cnt_iter < n_iters and error >= tol:
                # t0 = time.time()

                if self.batch_size is None:
                    Y = np.dot(X, w_new)

                    delta_w, delta_lambda, gradient = self.Newton_update(
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
                    delta_w, delta_lambda, gradient = self.Newton_update(
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

                if self.cnt_iter > 1:
                    error = w_similarity[self.cnt_iter]

                # Update the iterator
                if self.verbose:
                    pbar.update(1)
                    pbar.set_description(
                        f"Component {idx + 1} - " f"Error: {np.round(error, 6)}"
                    )

                # Update counter
                self.cnt_iter += 1

                # t1 = time.time()
                # print(f"Final steps: {t1 - t0} seconds")

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

                ax[7].set_title("Spectral constraint")
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

            # t1 = time.time()
            # print(f"Calculate metrics: {t1 - t0} seconds")

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # n = 512
            # freq = np.fft.fftfreq(n, 1 / 25e3)[: n // 2]
            # val = 2 / n * np.abs(np.fft.fft(w_new[:, 0]))[: n // 2]
            # ax.plot(freq, val)
            #
            # plt.show(block=True)

        # Return the updated matrices
        return W_, Lambda_

    def update_params(self, w_current, lambda_current, delta_w, delta_lambda, W, idx):
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
        approx_flag=False,
        Lambda=None,
        Fs: float | int = 25e3,
    ):
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
        n_iters=1,
        learning_rate=0.1,
        tol=1e-3,
        approx_flag=False,
        Fs: int | float = 25e3,
    ):
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
        self.spectral_obj = spectral_objective(
            m, save_hessian_flag=False, inv_hessian_flag=False, verbose=False
        )

        # Initialise the batch sampler
        if self.batch_size is not None:
            batch_sampler_inst = batch_sampler(self.batch_size, include_end=True)
            self._data_sampler = iter(batch_sampler_inst(X_preprocess, iter_idx=0))

        # Train based on the users choice (SUMT or standard)
        if self.sumt_flag:
            param_iters = []
            W_iters = [W.copy()]
            lambda_iters = [Lambda.copy()]
            cluster_info = []
            solution_error = []
            # penalty_error = [] Not applicable

            # Initialise the penalty parameter term
            self.alpha_reg = self.alpha_init

            while self.alpha_reg <= self.alpha_end:
                W_update, Lambda_update = self.spectral_fit(
                    X_preprocess,
                    W_iters[-1],
                    n_iters,
                    learning_rate,
                    tol,
                    approx_flag,
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

                # Get spectral_W
                spectral_W = self.compute_spectral_W(W_update)
                # self.cluster_components()

                # Store cluster info
                cluster_info.append((spectral_W,))

                solution_error.append(
                    np.mean(np.linalg.norm(W_iters[-1] - W_iters[-2], axis=1))
                )

                # Display the solution error
                if self.verbose:
                    print("----" * 10)
                    print(
                        f"\nFor alpha = {self.alpha_reg}, "
                        f"error = {solution_error[-1]}.\n"
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

            ## Check pos_min

            self.Lambda = lambda_iters[pos_min + 1]
            self.W = W_iters[
                pos_min + 1
            ]  # un-adjust pos_min as W_iters has an initial point.

            opt_cluster = cluster_info[pos_min]

            self.spectral_W = opt_cluster[0]

            # Store for looking at later.
            self.param_iters = param_iters
            self.lambda_iters = lambda_iters
            self.W_iters = W_iters
            self.cluster_info = cluster_info

        else:
            W_update, Lambda_update = self.spectral_fit(
                X_preprocess,
                W,
                n_iters,
                learning_rate,
                tol,
                approx_flag,
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
        return np.dot(self.preprocess_inst.preprocess_data(X), self.W.T)

    def inverse_transform(self, S):
        r, c = S.shape

        S_ = S.copy()

        # Transform back to the training feature space
        X_recon = np.dot(S_, self.W)

        # Un-process the data (still zero-mean, unit-variance)
        X_orig_recon = self.processor_inst.unprocess_data(X_recon)

        return X_orig_recon

    @staticmethod
    def compute_spectral_W(W):
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

    def get_solver_results(self):
        pass
