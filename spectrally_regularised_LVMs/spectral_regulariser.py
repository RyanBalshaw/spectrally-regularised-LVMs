# Copyright 2023-present Ryan Balshaw
"""
The spectral regularisation class used for LVM parameter estimation.
"""
import numpy as np


class SpectralObjective(object):
    """
    An object that implements the spectral constraint objective.

    Methods
    -------
    dftmtx()
            A method that returns the DFT matrix (unnormalised) by 1/sqrt(N).

    decompose_DFT()
            A method that decomposes the unnormalised DFT matrix into the real and
            imaginary matrices R and I.

    Hadamard_product(A, x)
            A method that computes the Hadamard product for the matrix-vector product
            v = A @ x. This returns v * v (elementwise product)

    Hadamard_derivative(A, x)
            A method that computes the Hadamard product derivative w.r.t the x
            vector. d/dx(v * v) = 2 * diag(A @ x) @ A

    check_w_want(w1)
            This method checks whether the vector that we compare to (w1) is suitable.
            It allows for either a Nx1 vector or a MxN matrix.

    Xw(Re, Im, w)
            This method computes the squared complex modulus of the Fourier
            vector F(w). The returned vector is normalised by 1/N. It accounts for
            the shape of w, to allow for a w to either be a Nx1 vector or a MxN matrix
            for M vectors w.

    spectral_loss(w, w1)
            This method returns the loss function for the spectral constraint.
            It computes the dot product between the Fourier representation of w
            and the Fourier representation/representations of the vector/vectors
            in w1. The resulting shape of the loss is either a 1x1 vector or a
            Mx1 vector (depending on whether w is a vector or matrix of vectors)

    spectral_derivative(w, w1)
            This method computes the first derivative (gradient) of the spectral loss
            w.r.t the w parameters. It returns a Nx1 vector of parameters
            or a NxM matrix of M gradient vectors.

    spectral_hessian(w, w1)
            This method computes the second derivative (Hessian) of the spectral loss
            w.r.t the w parameters. it either returns a NxN vector of parameters
            of a MxNxN matrix of M Hessians.
    """

    def __init__(self, N, save_hessian_flag=True, inv_hessian_flag=True, verbose=False):
        """
        This method initialises the spectral constraint. It allows the user to
        define whether they wish to save the Hessian matrix in the instance,
        whether the inverse hessian is to be calculated and returned, and
        defines whether the print statements are used.

        Parameters
        ----------
        N: int | float
                The dimensionality of the constraint domain.

        save_hessian_flag: bool
                A flag used to specify whether the hessian is saved locally in the
                instance after it has been calculated.

        inv_hessian_flag: bool
                A flag to specify whether the inverse hessian at computation time.

        verbose: bool
                A flag to control whether the print statements are used for the loss
                function, derivative and the hessian.

        Attributes
        ----------
        R: ndarray
                2D array of shape NxN contained data with 'float' type.
                The real components of the unnormalised DFT matrix.

        I: ndarray
                2D array of shape NxN contained data with 'float' type.
                The imaginary components of the unnormalised DFT matrix.
        """
        self.N = N
        self.save_hessian_flag = save_hessian_flag
        self.inv_hessian_flag = inv_hessian_flag
        self.verbose = verbose

        # Get R and I
        self.R, self.I = self.decompose_DFT()

    def dftmtx(self):
        """
        Computes the DFT matrix.

        Returns
        -------
        DFT_matrix: ndarray
                A 2D array of shape NxN contained data with 'complex' type. This is the
                unnormalised DFT matrix.
        """
        # # Method 1:
        # scipy.linalg.dft(N)

        # # Method 2:
        # L = np.arange(0, N, 1) #0 to N-1

        # omega_n = lambda factor: np.exp(-2 * np.pi * 1j /N * factor * L)

        # DFT_matrix = np.zeros((N, N), dtype = complex)

        # for i in range(N):
        #     DFT_matrix[:, i] = omega_n(i)

        # Method 3:
        omegas = np.exp(-2j * np.pi * np.arange(self.N) / self.N).reshape(-1, 1)
        DFT_matrix = omegas ** np.arange(self.N)

        return DFT_matrix

    def decompose_DFT(self):
        """
        Splits the complex DFT matrix into its real and imaginary components.

        Returns
        -------
        Re: ndarray
                2D array of shape NxN contained data with 'float' type.
                The real components of the unnormalised DFT matrix.

        Im: ndarray
                2D array of shape NxN contained data with 'float' type.
                The imaginary components of the unnormalised DFT matrix.
        """
        D = self.dftmtx()

        Re = np.real(D)
        Im = np.imag(D)

        return Re, Im

    @staticmethod
    def Hadamard_product(A, x, vectorised_flag=False):
        """
        A method that computes the Hadamard product of v = A @ x.

        Parameters
        ----------
        A: ndarray
                The matrix component of v of 'float' type.

        x: ndarray
                The vector component of v of 'float' type. Shape is either Nx1 or
                MxN

        vectorised_flag: bool
                A flag to specify whether x is a Nx1 vector or a MxN matrix.
                This is important to control whether v = A @ x (false) or
                v = x @ A.T (assuming that x is already transposed).

        Returns
        -------
                The elementwise product of v ʘ v.
        """
        # Ax = A @ x

        if vectorised_flag:
            v = x @ A.T

        else:
            v = A @ x

        return v**2  # (Ax) * (Ax) #

    @staticmethod
    def Hadamard_derivative(A, x):
        """
        A method that computes the derivative of Hadamard product of
        v = A @ x w.r.t x.

        This method has no requirement for a vectorised_flag variable as it is
        only to be called on the Fourier representation of the vector w, not
        the vector/matrix w1 that it is compared to.

        Parameters
        ----------
        A: ndarray
                The matrix component of v of 'float' type.

        x: ndarray
                The vector component of v of 'float' type. Shape is either Nx1 or
                MxN

        Returns
        -------
                The elementwise derivative of v ʘ v w.r.t x.
        """
        v = A @ x

        # Diagonalising is very slow! You can get the same output by just elementwise
        # multiplying A with the vector v.

        return 2 * v * A  # 2 * np.diag(Ax[:, 0]) @ A

    def check_w_want(self, w1):
        """
        A method that checks the shape of the vector w1.

        It is just used as a simple check to ensure that it matches the
        dimensionality of the problem

        Parameters
        ----------
        w1: npdarray
                The vector of interest

        Returns
        -------
        bool
                If w1, in some way, matches the dimensionality, it passes.
        """
        if w1.shape[0] == self.N and w1.shape[1] == 1:  # A Nx1 vector
            return True

        elif w1.shape[0] >= 1 and w1.shape[1] == self.N:  # A MxN matrix
            return True

        else:
            return False

    def Xw(self, Re, Im, w):
        """
        This method computes the squared modulus of the Fourier representation of
        a vector or matrix w. Instead of using a vectorised_flag (haramard_product is
        a staticmethod), I can just use the shape of w directly.

        Parameters
        ----------
        Re: ndarray
                2D array of shape NxN contained data with 'float' type.

        Im: ndarray
                2D array of shape NxN contained data with 'float' type.

        w: nparray
                2D array of shape Nx1 or MxN with 'float' type.

        Returns
        -------
        spectral_representation: ndarray
                The squared modulus of the Fourier transform of w. Shape is either
                Nx1 or MxN (depends on shape of w).
        """

        if w.shape[1] == 1:
            vec_flag = False

        else:
            vec_flag = True

        spectral_representation = (
            1
            / self.N
            * (
                self.Hadamard_product(Re, w, vec_flag)
                + self.Hadamard_product(Im, w, vec_flag)
            )
        )  # 1/N is included to satisfy Parseval's theorem (normalised)

        return spectral_representation

    def spectral_loss(self, w, w1):
        """
        This method computes the spectral constraint loss function.

        Parameters
        ----------
        w: ndarray
                The vector w that we wish to enforce is unique to the vector/vectors
                in w1. Expected shape is Nx1.

        w1: ndarray
                The vector/vectors that we wish to use to enforce that w is unique.
                Expected shape is Nx1 or MxN.

        Returns
        -------
        loss: float | ndarray
                The dot product between the spectral representations of the w vector
                and the vector/vectors in w1. It is either a scalar (if w1 is a vector)
                or a Mx1 vector (if w1 is a MxN vector).
        """

        if self.verbose:
            print("Determining the loss function...")

        try:
            assert self.check_w_want(w1)

        except AssertionError:
            print("W_want is the wrong shape!")
            raise SystemExit

        # Assume that w and w1 are column vectors
        # w = thing to optimise
        # w1 = goal
        # Calculate the loss
        # R, I = decompose_DFT(n)

        Xw_want = self.Xw(
            self.R, self.I, w1
        )  # np.convolve(w1[:, 0], w1[:, 0][::-1], mode = 'same').reshape(-1, 1))#

        # Xw_want = Xw_want * (Xw_want > 2 * np.std(Xw_want))

        Xw_current = self.Xw(
            self.R, self.I, w.reshape(-1, 1) if len(w.shape) == 1 else w
        )

        if w1.shape[1] == 1:
            loss = Xw_want.T @ Xw_current
            return loss[0, 0]

        else:  # w1 is a k x N matrix of k vectors
            loss = Xw_want @ Xw_current
            return loss

    def spectral_derivative(self, w, w1):
        """
        This method computes the first derivative of the spectral constraint
        loss function w.r.t w.

        Parameters
        ----------
        w: ndarray
                The vector w that we wish to enforce is unique to the vector/vectors
                in w1. Expected shape is Nx1.

        w1: ndarray
                The vector/vectors that we wish to use to enforce that w is unique.
                Expected shape is Nx1 or MxN.

        Returns
        -------
        gradient: ndarray
                The first derivative of the dot product between the spectral
                representations of the w vector and the vector/vectors in w1.
                It is either a Nx1 vector (if w1 is a vector) or a NxM matrix
                (if w1 is a MxN vector).

                Note: I chose to use a NxM matrix for the latter as I know
                each column represents a gradient vector as the constraint is applied
                additively, so I can just sum over axis=1 to get a combined gradient
                vector.
        """

        if self.verbose:
            print("Determining the gradient...")

        try:
            assert self.check_w_want(w1)

        except AssertionError:
            print("W_want is the wrong shape!")
            raise SystemExit

        # n = w.shape[0]
        # R, I = decompose_DFT(n)

        # Calculate the derivative

        Xw_want = self.Xw(
            self.R, self.I, w1
        )  # np.convolve(w1[:, 0], w1[:, 0][::-1], mode = 'same').reshape(-1, 1))#
        # Xw_want = Xw_want * (Xw_want > 2 * np.std(Xw_want))

        term1 = self.Hadamard_derivative(
            self.R, w.reshape(-1, 1) if len(w.shape) == 1 else w
        )
        term2 = self.Hadamard_derivative(
            self.I, w.reshape(-1, 1) if len(w.shape) == 1 else w
        )

        if w1.shape[1] == 1:
            gradient = (
                (1 / self.N) * Xw_want.T @ (term1 + term2)
            )  # normalise by 1/N to account for term not included in (term1 + term2)

            return gradient.T[:, 0]

        else:
            gradient = (
                (1 / self.N) * Xw_want @ (term1 + term2)
            )  # normalise by 1/N to account for term not included in (term1 + term2)

            return gradient.T  # See note in docstring

    def spectral_hessian(self, w, w1):
        """
        This method computes the second derivative of the spectral constraint
        loss function w.r.t w.

        Parameters
        ----------
        w: ndarray
                The vector w that we wish to enforce is unique to the vector/vectors
                in w1. Expected shape is Nx1.

        w1: ndarray
                The vector/vectors that we wish to use to enforce that w is unique.
                Expected shape is Nx1 or MxN.

        Returns
        -------
        hessian: ndarray
                The second derivative of the dot product between the spectral
                representations of the w vector and the vector/vectors in w1.
                It is either a NxN matrix (if w1 is a vector) or a MxNxN
                matrix (if w1 is a MxN vector).

        If save_hessian_flag is used during initialisation, it will first
        check to see if a Hessian exists.

        If inv_hessian_flag is used during initialisation, it will return
        both the hessian and its inverse (NOT recommended).
        """

        try:
            assert self.check_w_want(w1)

        except AssertionError:
            print("W_want is the wrong shape!")
            raise SystemExit

        # n = w.shape[0]
        # R, I = decompose_DFT(n)

        # Calculate the Hessian

        if not hasattr(self, "hessian") or not self.save_hessian_flag:
            if self.verbose:
                print("Determining the hessian...")

            Xw_want = self.Xw(
                self.R, self.I, w1
            )  # np.convolve(w1[:, 0], w1[:, 0][::-1], mode = 'same').reshape(-1, 1))#
            # Xw_want = Xw_want * (Xw_want > 2 * np.std(Xw_want))

            # One approach to remove the for loop
            # dir_mat = R.reshape(-1, 1) * np.repeat(R, n, axis = 0) +
            # I.reshape(-1, 1) * np.repeat(I, n, axis = 0)
            # dir_mat = dir_mat.reshape(n, n, n).transpose(2, 0, 1)

            if not hasattr(self, "dir_mat"):  # compute it once off
                dir_mat = np.zeros((self.N, self.N, self.N))

                for c in range(self.N):
                    # term1 = np.diag(self.R[:, r]) @ self.R
                    # term2 = np.diag(self.I[:, r]) @ self.I

                    dir_mat[c, :, :] = self.R[:, [c]] * self.R + self.I[:, [c]] * self.I

                self.dir_mat = dir_mat

            if w1.shape[1] == 1:
                hessian = (2 / self.N) * (
                    np.tensordot(Xw_want.T, self.dir_mat, axes=([1], [1]))[0, :, :]
                )  # normalise by 2/N to account for term not included in (dir_mat)

            else:
                hessian = (2 / self.N) * np.tensordot(
                    Xw_want, self.dir_mat, axes=([1], [1])
                )  # normalise by 2/N to account for term not included in (dir_mat)

            # hessian = 2 * Xw_want.T @ dir_mat.transpose(1, 0, 2)
            # hessian = 2 * hessian.transpose(1, 0, 2)

            if self.save_hessian_flag:
                self.hessian = hessian

            return hessian

        elif self.inv_hessian_flag and self.save_hessian_flag:
            self.inv_hessian = np.linalg.inv(self.hessian)  # sorry Prof. Kok

            return self.hessian, self.inv_hessian

        else:
            return self.hessian
