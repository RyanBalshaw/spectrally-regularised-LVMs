"""
The helper methods for the spectrally constrained FastICA solver.

Method 1: ica_batch_sampler
    This is a simple batch sampler method for quicker solving.

Method 2: quasi_Newton
    This is a method that implements different Hessian approximation strategies.
"""

import numpy as np


class ica_batch_sampler(object):
    """
    This is a simple iterator instance that can be called during runtime using:

    batch_sampler = ica_batch_sampler(batch_size, include_end=True)
    data_sampler = iter(batch_sampler(X_preprocess, iter_idx=0))

    and then sampling in a loop or something like that

    Xi = next(data_sampler)
    """

    def __init__(self, batch_size, random_sampler=True, include_end=False):
        """
        Parameters
        ----------
        batch_size: int
            The batch size used by the sampler.

        random_sampler: bool
            A flag to specify whether the sampler runs by randomly selecting indices
            from the data (random_sampler=True) or if it loops through
            the data in sequence.

        include_end: bool
            A flag to specify whether the data that does not fit into integer
            batch increments is used.
        """
        self.batch_size = batch_size
        self.random_sampler = random_sampler
        self.include_end = include_end

    def __call__(self, data, iter_idx=0):
        """
        The method used when the sampler instance is called (like a function).

        Parameters
        ----------
        data: ndarray
            A matrix of data samples.

        iter_idx: int
            This specifies which axis in data is to be iterated over.

        Returns
        -------
        self

        Raises
        ------
        AssertionError
            This is raised when the data is not a numpy array

        """
        if hasattr(data, "copy"):
            self._data = (
                data.copy()
            )  # Implicity requires the data to have a copy method

        else:
            print(
                "What kind of data are we iterating over? "
                "It does not have a copy method."
            )
            self._data = data

        try:
            assert type(self._data).__module__ == np.__name__

        except AssertionError:
            print("Data is not a numpy instance.")
            raise SystemExit

        self._iter_idx = iter_idx

        if self.random_sampler:
            self._max_iter = np.inf

        else:
            self._iter_range = list(
                range(0, self._data.shape[self._iter_idx], self.batch_size)
            )

            if self.include_end:
                self._iter_range.append(self._data.shape[self._iter_idx])

            self._max_iter = len(self._iter_range) - 1

        return self

    def __iter__(self):
        """
        Initalises the iteration counter when iter() is applied to the sampler
        instance.

        Returns
        -------
        self
        """
        self._iter_cnt = 0

        return self

    def __getitem__(self, idx):
        """
        I cannot remember what this is for.
        Perhaps it is redundant?

        Parameters
        ----------
        idx: int
            The index of the iteration indices that are extracted.

        Returns
        -------
            The starting point of the iteration range.
        """
        return self._iter_range[idx]

    def __next__(self):
        """
        This is called during for looping over the iterator or using next(iterator).

        Returns
        -------
        data_batch: ndarray
            The samples from the observed data.

        """
        if self._iter_cnt < self._max_iter:
            pass

        else:
            self._iter_cnt = 0
            # raise StopIteration # I am breaking many Python laws.

        # This code should be in the if statement
        if not self.random_sampler:
            idx_l = self._iter_range[self._iter_cnt]
            idx_u = self._iter_range[self._iter_cnt + 1]
            data_batch = np.take(
                self._data, range(idx_l, idx_u, 1), axis=self._iter_idx
            )

        else:
            data_batch = np.take(
                self._data,
                np.random.randint(0, self._data.shape[self._iter_idx], self.batch_size),
                axis=self._iter_idx,
            )

        self._iter_cnt += 1

        return data_batch


class quasi_Newton(object):
    """
    This method implements different Hessian approximation strategies and
    performs the updates on each call.

    The included quasi-Newton methods are:
    - Symmetric rank one (SR1)
    - Davidson Fletcher Powell (DFP)
    - Boyden-Fletcher-Goldfarb-Shanno (BFGS)

    Each method accepts the delta_x are each iteration index and
    the delta_grad. These are the two attributes typically used for
    quasi-Newton iteration.
    """

    def __init__(
        self,
        jacobian_update_type: str,
        use_inverse: bool = True,
    ):
        """
        Parameters
        ----------
        jacobian_update_type: str
            Specifies the quasi-Newton method used.

        use_inverse: bool
            A flag used to specify whether the Hessian inverse or the
            standard Hessian is to be approximated.
        """
        self.jacobian_update_type = jacobian_update_type.lower()  # SR1, DFP, BFGS
        self.use_inverse = use_inverse
        self.iter_index = 0

    def symmetric_rank_one(self, delta_params_k, grad_diff_k):
        """
        The SR1 update step

        Parameters
        ----------
        delta_params_k: ndarray
            The parameter difference (x_{t} - x_{t-1}) vector.

        grad_diff_k: ndarray
            The gradient difference (df/dx @ x_{t} - df/dx @ x_{t-1}) vector.

        Returns
        -------
        update_term: ndarray
            The SR1 update step factoring in whether the inverse Hessian or direct
            Hessian are approximated.
        """
        if self.use_inverse:
            t1 = delta_params_k - self.jacobian_mat_iter @ grad_diff_k

            update_term = (t1 @ t1.T) / (t1.T @ grad_diff_k)

        else:
            t1 = grad_diff_k - self.jacobian_mat_iter @ delta_params_k

            update_term = (t1 @ t1.T) / (t1.T @ delta_params_k)

        return update_term

    def davidson_fletcher_powell(self, delta_params_k, grad_diff_k):
        """
        The DFP update step

        Parameters
        ----------
        delta_params_k: ndarray
            The parameter difference (x_{t} - x_{t-1}) vector.

        grad_diff_k: ndarray
            The gradient difference (df/dx @ x_{t} - df/dx @ x_{t-1}) vector.

        Returns
        -------
        update_term: ndarray
            The DFP update step factoring in whether the inverse Hessian or direct
            Hessian are approximated.
        """
        gamma_k = 1 / (grad_diff_k.T @ delta_params_k)

        if self.use_inverse:
            t1 = delta_params_k @ delta_params_k.T * gamma_k
            t2 = (
                -1
                * self.jacobian_mat_iter
                @ grad_diff_k
                @ grad_diff_k.T
                @ self.jacobian_mat_iter
                / (grad_diff_k.T @ self.jacobian_mat_iter @ grad_diff_k)
            )

            update_term = t1 + t2

        else:
            I_mat = np.eye(grad_diff_k.shape[0])

            t1 = I_mat - grad_diff_k @ delta_params_k.T * gamma_k
            t2 = I_mat - delta_params_k @ grad_diff_k.T * gamma_k
            t3 = grad_diff_k @ grad_diff_k.T * gamma_k

            update_term = (
                (t1) @ self.jacobian_mat_iter @ (t2) + t3 - self.jacobian_mat_iter
            )  # subtract to fix update rule below

        return update_term

    def boyden_fletcher_goldfarb_shanno(self, delta_params_k, grad_diff_k):
        """
        The BFGS update step

        Parameters
        ----------
        delta_params_k: ndarray
            The parameter difference (x_{t} - x_{t-1}) vector.

        grad_diff_k: ndarray
            The gradient difference (df/dx @ x_{t} - df/dx @ x_{t-1}) vector.

        Returns
        -------
        update_term: ndarray
            The BFGS update step factoring in whether the inverse Hessian or direct
            Hessian are approximated.
        """
        gamma_k = 1 / (grad_diff_k.T @ delta_params_k)

        if self.use_inverse:
            I_mat = np.eye(grad_diff_k.shape[0])

            t1 = (
                I_mat - (delta_params_k @ grad_diff_k.T) * gamma_k
            )  # grad_diff_k.T @ self.jacobian_mat_iter @ grad_diff_k
            t2 = (
                I_mat - (grad_diff_k @ delta_params_k.T) * gamma_k
            )  # delta_params_k.T @ grad_diff_k
            t3 = delta_params_k @ grad_diff_k.T * gamma_k

            update_term = (
                t1 @ self.jacobian_mat_iter @ t2 + t3 - self.jacobian_mat_iter
            )  # subtract to fix update rule below

        else:
            t1 = (grad_diff_k @ grad_diff_k.T) / (grad_diff_k.T @ delta_params_k)
            t2 = (
                self.jacobian_mat_iter
                @ delta_params_k
                @ delta_params_k.T
                @ self.jacobian_mat_iter.T
            ) / (delta_params_k.T @ self.jacobian_mat_iter @ delta_params_k)

            update_term = t1 - t2

        return update_term

    def initialise_jacobian(self, m):
        """
        A method that initialises the Jacobian matrix.
        Assumes that

        Parameters
        ----------
        m: int
            The dimensionality of the feature space.

        Initialises
        -----------
        self.jacobian_mat_iter: ndarray
            The Hessian used during iteration.
        """
        if self.use_inverse:
            if self.jacobian_update_type == "bfgs":
                self.jacobian_mat_iter = np.eye(m)

            else:
                self.jacobian_mat_iter = 0.1 * np.eye(m)
        else:
            if self.jacobian_update_type == "bfgs":
                self.jacobian_mat_iter = np.eye(m)

            else:
                self.jacobian_mat_iter = 10 * np.eye(m)

    def compute_update(self, gradient_vector):
        """
        A method used to compute the parameter update
        based on the gradient vector at time t.

        Parameters
        ----------
        gradient_vector: ndarray
            The Nx1 gradient vector

        Returns
        -------
        update: ndarray
            The quasi-Newton parameter update.
        """
        if not hasattr(self, "jacobian_mat_iter"):
            self.initialise_jacobian(gradient_vector.shape[0])

        if self.use_inverse:
            update = -1 * self.jacobian_mat_iter @ gradient_vector

        else:
            update = -1 * np.linalg.solve(self.jacobian_mat_iter, gradient_vector)

        return update

    def update_jacobian(self, delta_params_k, grad_diff_k):
        """
        A method that updates the jacobian_mat_iter attribute.

        Parameters
        ----------
        delta_params_k: ndarray
            The parameter difference (x_{t} - x_{t-1}) vector.

        grad_diff_k: ndarray
            The gradient difference (df/dx @ x_{t} - df/dx @ x_{t-1}) vector.
        """
        if self.jacobian_update_type == "sr1":
            update_term = self.symmetric_rank_one(delta_params_k, grad_diff_k)

        elif self.jacobian_update_type == "dfp":
            update_term = self.davidson_fletcher_powell(delta_params_k, grad_diff_k)

        elif self.jacobian_update_type == "bfgs":
            update_term = self.boyden_fletcher_goldfarb_shanno(
                delta_params_k, grad_diff_k
            )

        else:
            print(f"Update type ({self.jacobian_update_type}) not understood.")
            raise SystemExit

        self.jacobian_mat_iter += update_term
        self.iter_index += 1
