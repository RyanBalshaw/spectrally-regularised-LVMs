"""
The helper methods for the spectrally constrained FastICA solver.

Method 1: ica_batch_sampler
    This is a simple batch sampler method for quicker solving.

Method 2: quasi_Newton
    This is a method that implements different Hessian approximation strategies.
"""

import copy

import numpy as np


class data_processor(object):
    """
    A method that processes the data matrices.

    Methods
    -------
    initialise_preprocessing(X)
        A method that initialises all of the processing attributes
        for the pre-processing (standardising with whitening). Solves
        for the whitening transform parameters.

    standardise_data(X)
        Standardises the columns of X to be zero-mean and unit-variance.

    preprocess_data(X)
        Transforms the X matrix to the whitened space (if required).

    unprocess_data(X)
        Transforms a X matrix from the whitened space to the data space.

    """

    def __init__(self, whiten: bool = True, var_PCA: float | None = None):
        """
        Parameters
        ----------
        whiten: bool
            A flag to control whether the processing should whiten the data.

        var_PCA: float | None (default None)
            A value that is in [0, 1] that specifies how much of the variance
            you wish to keep. Allows one to remove some of the non-dominant sources
            in the data.
        """
        self.whiten = whiten
        self.var_PCA = var_PCA

    def initialise_preprocessing(self, X):
        """
        A method that initialises the aspects of the data pre-processing stage.

        Parameters
        ----------
        X: ndarray
            The initialisation matrix from which the pre-processing parameters are
            obtained.

        Returns
        -------
        self
        """

        # Extract Nsamples and Nfeatures
        Ns, Nf = X.shape

        # Get the mean and standard deviation (always automatically z-score)
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)

        if self.whiten:
            # Decompose X
            # rows of Vh are the eigenvectors of A^H A (i.e., Vh = U^T)
            # Columns of U are eigenvectors of A A^H
            U, s, Vh = np.linalg.svd(
                (copy.deepcopy(X) - self.mean_) / self.std_, full_matrices=False
            )

            eigenvalues = s**2 / (Ns - 1)

            if self.var_PCA is not None:
                # TODO improve var_PCA option

                cumsum = np.cumsum(eigenvalues / np.sum(eigenvalues))

                n_comp = np.argmin(np.abs(cumsum - self.var_PCA))

                self.Vh = Vh[:n_comp, :]
                self.eigenvalues = eigenvalues[:n_comp]

            else:
                self.Vh = Vh
                self.eigenvalues = eigenvalues

            # Whiten the data
            L_inv_sqrt = np.diag(1 / np.sqrt(self.eigenvalues))
            L_sqrt = np.diag(np.sqrt(self.eigenvalues))

            self.latent_transform = np.dot(L_inv_sqrt, self.Vh)
            self.recover_transform = np.dot(L_sqrt, self.Vh)

        else:
            self.Vh = np.eye(Nf)
            self.latent_transform = np.eye(Nf)
            self.recover_transform = np.eye(Nf)

        return self

    def standardise_data(self, X):
        """
        A method that standardises the row of the data matrix X

        Parameters
        ----------
        X: ndarray
            original feature matrix

        Returns
        -------
        X_standardised: ndarray
            Zero-mean, unit variance feature matrix.
        """

        X_standardised = (X - self.mean_) / self.std_

        return X_standardised

    def preprocess_data(self, X):
        """
        A method that pre-processes the data matrix X

        Parameters
        ----------
        X: ndarray
            original feature matrix

        Returns
        -------
        X_whitened: ndarray
            The whitened feature matrix
        """
        X_standardised = self.standardise_data(X)

        X_whitened = (X_standardised @ self.latent_transform.T) @ self.Vh
        # X_whitened = np.dot(X_standardised, self.latent_transform.T)

        return X_whitened

    def unprocess_data(self, X):
        """
        A method that unwhitens the whitened data matrix X

        Parameters
        ----------
        X: ndarray
            The whitened feature matrix

        Returns
        -------
        X_unwhitened: ndarray
            The unwhitened feature matrix
        """
        X_unwhitened = np.dot(np.dot(X, self.recover_transform.T), self.Vh)
        # X_unwhitened = np.dot(X, self.recover_transform.T)

        return X_unwhitened


class batch_sampler(object):
    """
    This is a simple iterator instance that can be called during runtime using:

    batch_sampler_inst = batch_sampler(batch_size, include_end=True)
    data_sampler = iter(batch_sampler_inst(X_preprocess, iter_idx=0))

    and then sampling in a loop or something like that:

    for i in range(10):
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


class deflation_orthogonalisation(object):
    """
    This method implements the Gram-Schmidt orthogonalisation
    process and some helper methods.

    Methods
    -------
    projection_operator(u, v)
        words

    gram_schmidt_orthogonalisation(w, W, idx)
        words

    global_gso(W)
        words

    """

    @staticmethod
    def projection_operator(u, v):
        """
        Calculates projection of v onto u (equivalent to outer product map).

        Parameters
        ----------
        u: ndarray
            A Nx1 array that we wish to orthogalise against (remains unchanged)
        v: ndarray
            A Nx1 array that we wish to orthogonalise (changes)

        Returns
        -------

        """

        return u.T @ v / (u.T @ u) * u

    def gram_schmidt_orthogonalisation(self, w, W, idx):  # Important to FastICA
        """

        Parameters
        ----------
        w: ndarray
            A Nx1 array that contains the vector we want to orthogonalise.

        W: ndarray
            A MxN array that contains the vectors we want to orthogonalise w against.

        idx: int
            The upper index (cannot be zero) for the rows of W that
            we want to orthogonalise against.

        Returns
        -------
        w_orth: ndarray
            A Nx1 array that contains the orthogonalised w vector using the first
            idx + 1 vectors in W.

        Note: I have played around with vectorised versions of this, but
        it did not offer significant computational improvements.
        """
        # W is a matrix (w_1, ..., w_N)^T of already orthogonalised vectors.
        # idx is the index of the vectors in
        # W (w_1, ..., w_idx) that we wish to orthogonalise w against
        # Note: idx cannot be zero

        w_orth = w.copy()

        if idx == 0:
            print("GSO index cannot be zero.")
            raise SystemExit

        if idx > W.shape[0]:
            print(
                "GSO index exceeds the number of vectors you want to compare against."
            )
            raise SystemExit

        # Perform Gram-Schmidt Orthogonalisation
        for i in range(0, idx, 1):
            w_orth = w_orth - self.projection_operator(W[[i], :].T, w_orth)

        # Normalise w_gorth
        w_orth /= np.linalg.norm(w_orth)

        return w_orth

    def global_gso(self, W):  # Important to FastICA
        """
        A method that orthogonalises a set of Nx1 vectors
        stores in some W matrix of shape MxN.

        Parameters
        ----------
        W: ndarray
            A MxN array that contains the vectors we want to orthogonalise
            in the rows (i.e. assumes that W = [w_1, w_N]^T).

        Returns
        -------
        W_orth: ndarray
            A MxN array of orthogonalised vectors.
        """

        W_orth = np.zeros_like(W, dtype="f")

        # Make the first vector unit (do not change)
        W_orth[0, :] = W[0, :] / np.linalg.norm(W[0, :])

        # Iterate over the other vectors and orthogonalise against them
        for i in range(1, W.shape[0], 1):
            W_orth[i, :] = self.gram_schmidt_orthogonalisation(W[[i], :].T, W, i)[:, 0]

        return W_orth


def Hankel_matrix(signal, Lw=512, Lsft=1):
    """
    A method that performs hankelisation for the user.

    Parameters
    ----------
    signal: ndarray
        A (n,) shaped array that contains a time series of measurement values

    Lw: int
        The window length/signal segment length

    Lsft: int
        The shift parameter for the sliding window

    Returns
    -------
    Hmat: ndarray
        A no_of_samples x Lw array of sliding window segments.

    """
    signal = signal.flatten()
    N = len(signal)

    # Pre-allocate the size of Hmat
    no_of_samples = int(np.floor((N - Lw) / Lsft) + 1)

    # Initialise
    Hmat = np.zeros((no_of_samples, Lw))

    # Store the segments
    for i in range(no_of_samples):
        start = int(i * Lsft)
        end = int(Lw + i * Lsft)

        Hmat[i, :] = signal[start:end]

    return Hmat
