# Copyright 2023-present Ryan Balshaw
"""
This set of methods that define general cost functions that
one can use, and two specific methods (principal component analysis
and independent component analysis).
"""
import numpy as np
import sympy as sp

from .negen_approx import initialise_sources


class costClass(object):
    """
    Base class for different formulations of the user
    cost function.

    Finish write up here!
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def set_cost(self, cost_func):
        """
        This method allows one to set their cost function.

        Parameters
        ----------
        cost_func : function
            The users cost function.
            Example: cost_func = lambda X, w, y: -1 * np.mean(y ** 2, axis=0)

        """
        self._cost = cost_func

    def set_gradient(self, cost_gradient):
        """
        This method allows one to set their gradient vector.

        Parameters
        ----------
        cost_gradient : function
            The users gradient vector of the cost function.
        Example:
            cost_gradient = lambda X, w, y: -2 * np.mean(y * X, axis=0, keepdims = True)

        """
        self._cost_gradient = cost_gradient

    def set_hessian(self, cost_hessian):
        """
        This method allows one to set their objective Hessian (optional).

        Parameters
        ----------
        cost_hessian : function
            The users gradient vector of the cost function.
            Example: cost_hessian = lambda X, w, y: -2 /X.shape[0] * (X.T @ X)

        """
        self._cost_hessian = cost_hessian

    def get_cost(self):
        """
        Method to return the cost function to the user.

        Returns
        -------
        cost_func if attribute exists, else None.
        """
        if hasattr(self, "_cost"):
            return self._cost
        else:
            return None

    def get_gradient(self):
        """
        Method to return the derivative function to the user.

        Returns
        -------
        cost_gradient if attribute exists, else None.
        """
        if hasattr(self, "_cost_gradient"):
            return self._cost_gradient
        else:
            return None

    def get_hessian(self):
        """
        A method to return the hessian function to the user.

        Returns
        -------
        cost_hessian if attribute exists, else None.
        """
        if hasattr(self, "_cost_hessian"):
            return self._cost_hessian
        else:
            return None

    def check_gradient(self, X, w, y, step_size):
        # Finite difference gradient approximation (central difference)
        # Method cannot work without
        # NB! - self._cost must be a primary function of X and w, not y

        if self.verbose:
            print("\nChecking the gradient using central difference approximation...")
        w0 = w.copy().reshape(-1, 1)

        grad_current = self.cost_gradient(X, w0, y).reshape(-1, 1)
        grad_check = np.zeros_like(grad_current)

        for i in range(grad_check.shape[0]):
            e_i = np.zeros_like(w0)
            e_i[i, 0] = step_size

            f_f = self.cost(X, w0 + e_i, y)
            f_b = self.cost(X, w0 - e_i, y)

            grad_check[i, 0] = (f_f - f_b) / (2 * step_size)

        grad_norm = np.linalg.norm(grad_current - grad_check)

        if self.verbose:
            print(f"Finished! The gradient norm is: {np.round(grad_norm)}")

        return grad_current, grad_check, grad_norm

    def check_hessian(self, X, w, y, step_size):
        # Finite difference Hessian approximation (central difference)

        if self.verbose:
            print("\nChecking the hessian using central difference approximation...")

        w0 = w.copy().reshape(-1, 1)
        hess_current = self.cost_hessian(X, w0, y)
        hess_check = np.zeros_like(hess_current)

        r, c = hess_check.shape

        for i in range(r):
            e_i = np.zeros_like(w0)
            e_i[i, 0] = step_size

            for j in range(c):
                e_j = np.zeros_like(w0)
                e_j[j, 0] = step_size

                f1 = self.cost(X, w0 + e_i + e_j, y)
                f2 = self.cost(X, w0 + e_i - e_j, y)
                f3 = self.cost(X, w0 - e_i + e_j, y)
                f4 = self.cost(X, w0 - e_i - e_j, y)

                hess_check[i, j] = (f1 - f2 - f3 + f4) / (4 * step_size**2)

        hess_norm = np.mean(np.linalg.norm(hess_current - hess_check, axis=1))

        if self.verbose:
            print(f"Finished! The hessian norm (row-wise) is: {np.round(hess_norm)}")

        return hess_current, hess_check, hess_norm


class user_cost(costClass):
    """
    An object that implements the a general user cost function
     class. This allows the user to manually define their cost function
     and associated gradient vector and hessian.

     Assumed format for methods: func(X, w, y) where X is an ndarray
      with shape (n_samples, n_features), w is an ndarray with shape (n_features, 1)
      and y is the linear transformation X @ w with shape (n_samples, 1).

    """

    def __init__(self, use_hessian: bool = True, verbose: bool = True):
        """

        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the solver, as you may wish to just
            perform steepest descent instead of using Newton's method.
        """
        super().__init__(verbose)
        self.use_hessian = use_hessian

    def cost(self, X, w, y):
        """
        A method that returns the cost function for the inputs.

        Parameters
        ----------
        X : ndarray
            The feature matrix of size (n_samples, n_features)
        w : ndarray
            The transformation vector of size (n_features, 1)

        y : ndarray
            The transformed variable y = X @ w of size (n_samples, 1)

        Returns
        -------
            cost function evalation of (X, w, y)
        """
        return self._cost(X, w, y)

    def cost_gradient(self, X, w, y):
        """
        A method that returns the cost function gradient for the inputs.

        Parameters
        ----------
        X : ndarray
            The feature matrix of size (n_samples, n_features)
        w : ndarray
            The transformation vector of size (n_features, 1)

        y : ndarray
            The transformed variable y = X @ w of size (n_samples, 1)

        Returns
        -------
            derivative function evalation of (X, w, y)
        """
        return self._cost_gradient(X, w, y)

    def cost_hessian(self, X, w, y):
        """
        A method that returns the Hessian function for the inputs.

        Parameters
        ----------
        X : ndarray
            The feature matrix of size (n_samples, n_features)
        w : ndarray
            The transformation vector of size (n_features, 1)

        y : ndarray
            The transformed variable y = X @ w of size (n_samples, 1)

        Returns
        -------
            Hessian function evalation of (X, w, y)
        """
        if self.use_hessian:
            return self._cost_hessian(X, w, y)

        else:
            return np.eye(w.shape[0])


class sympy_cost(costClass):
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        use_hessian: bool = False,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.n_samples = n_samples
        self.n_features = n_features
        self.use_hessian = use_hessian

    def set_cost(self, cost_func):
        # overwrites the base method
        """
        This method allows one to set their cost function (overwrites default).

        Parameters
        ----------
        cost_func : function
            The users cost function.
            Example: cost_func = lambda X, w, y: -1 * np.mean(y ** 2, axis=0)

        """
        self._sympy_cost = cost_func

    def get_model_parameters(self):
        i, j = sp.symbols("i j", cls=sp.Idx)

        self.w = sp.IndexedBase("w", shape=(self.n_features, 1))
        self.X = sp.IndexedBase("X", shape=(self.n_samples, self.n_features))
        self.y = sp.symbols("y")  # Placeholder variable

        return self.X, self.w, (i, j)

    def implement_cost(self):
        print("Lambdifying the sympy cost function...")

        self._cost = sp.lambdify(
            (self.X, self.w, self.y), self._sympy_cost
        )  # Will overwrite the sympy variable

    def implement_first_derivative(self):
        print("Deriving and lambdifying the sympy derivative function...")

        self._first_derivative_sympy = sp.Matrix(
            [self._sympy_cost.diff(self.w[i, 0]) for i in range(self.n_features)]
        )
        self._cost_gradient = sp.lambdify(
            (self.X, self.w, self.y), self._first_derivative_sympy
        )

    def implement_second_derivative(self):
        print("Deriving and lambdifying the sympy Hessian function...")

        if self.use_hessian:
            self._second_derivative_sympy = sp.BlockMatrix(
                [
                    self._first_derivative_sympy.diff(self.w[i, 0])
                    for i in range(self.n_features)
                ]
            )
            self._cost_hessian = sp.lambdify(
                (self.X, self.w, self.y), self._second_derivative_sympy
            )

        else:
            self._cost_hessian = lambda X, w, y: np.eye(w.shape[0])

    def implement_methods(self):
        if hasattr(self, "_sympy_cost"):
            print("Calculating the sympy method components...")

            self.implement_cost()
            self.implement_first_derivative()

            if self.use_hessian:
                self.implement_second_derivative()

        else:
            print(
                "Please first initialise the sympy cost function "
                "using inst.set_cost()."
            )
            raise SystemExit

    def check_X(self, X):
        r, c = X.shape
        assert r == self.n_samples, print(
            f"Number of samples ({r}) in X  does not match the"
            f" expected value: {self.n_samples}."
        )
        assert c == self.n_features, print(
            f"Number of features ({c}) in X does not match the"
            f" expected value: {self.n_features}."
        )

    def check_w(self, w):
        r, c = w.shape
        assert r == self.n_features, print(
            f"Number of samples ({r}) in w  does not match the"
            f" expected value: {self.n_features}."
        )
        assert c == 1, print(
            f"Number of features ({c}) in w does not match the" f" expected value: {1}."
        )

    def cost(self, X, w, y, *args):
        self.check_X(X)
        self.check_w(w)

        if not hasattr(self, "_cost"):
            self.implement_methods()  # Ensure creation of _methods.

        return self._cost(X, w, y)

    def cost_gradient(self, X, w, y, *args):
        return self._cost_gradient(X, w, y)

    def cost_hessian(self, X, w, y, *args):
        if self.use_hessian:
            return self._cost_hessian(X, w, y)

        else:
            return np.eye(w.shape[0])


class negentropy_cost(costClass):
    def __init__(self, source_name: str, source_params: dict, verbose: bool = True):
        super().__init__(verbose)
        self.source_name = source_name
        self.source_params = source_params  # dictionary of parameters

        # Initialise the source PDFs
        self.source_instance, self.source_expectation = initialise_sources(
            source_name, self.source_params
        )

    def cost(self, X, w, y):  # Important to negentropy-based ICA
        # Negentropy-estimate calculation

        if y.shape[1] == 1:
            EG_y = np.mean(self.source_instance.function(y))

        else:
            EG_y = np.mean(self.source_instance.function(y), axis=0)

        return -1 * (EG_y - self.source_expectation) ** 2

    def cost_gradient(self, X, w, y):  # Important to negentropy-based ICA
        N, m = X.shape

        g_y = self.source_instance.first_derivative(y)

        # Calculate the expectation
        expectation = np.mean(g_y * X, axis=0, keepdims=True).T

        # Calculate the derivative scale with the missing term
        r = np.mean(self.source_instance.function(y)) - self.source_expectation

        # Calculate the gradient vector
        grad_vector = -2 * r * expectation

        return grad_vector

    def cost_hessian(
        self, X, w, y, approx_flag=True
    ):  # Important to negentropy-based ICA
        N, m = X.shape

        # Compute g'(y)
        g_prime_y = self.source_instance.second_derivative(y)

        # t0 = time.time()
        if approx_flag:
            # t0 = time.time()
            expectation = np.mean(g_prime_y) * np.eye(m)
            # t1 = time.time()

        else:
            expectation = np.zeros((m, m))

            for n in range(N):
                expectation += np.dot(X[[n], :].T, X[[n], :]) * g_prime_y[n]

            expectation /= N

        # Calculate the gradient vector
        negentropy_gradient = self.cost_gradient(X, w, y)

        # Calculate the scalar r term
        r = np.mean(self.source_instance.function(y)) - self.source_expectation

        # Calculate the gradient outer product
        grad_outer = negentropy_gradient @ negentropy_gradient.T

        # Calculate the Jacobian (Hessian)
        jacobian = -2 * (grad_outer + r * expectation)

        return jacobian


class variance_cost(user_cost):
    def __init__(self, use_hessian: bool = True, verbose: bool = True):
        super().__init__(use_hessian, verbose)

        def loss(X, w, y):
            return -1 * np.mean((X @ w) ** 2, axis=0)

        def grad(X, w, y):
            return -2 * np.mean(y * X, axis=0, keepdims=True).T

        def hess(X, w, y):
            return -2 * np.cov(X, rowvar=False)

        self.set_cost(loss)
        self.set_gradient(grad)
        self.set_hessian(hess)


# if __name__ == "__main__":
#
#     n_samples = 10000
#     n_features = 3
#
#     # Method 1
#     test_inst1 = sympy_cost(n_samples, n_features, use_hessian=True)
#
#     X, w, iter_params = test_inst1.get_model_parameters()
#     i, j = iter_params
#
#     loss_i = sp.Sum(w[j, 0] * X[i, j], (j, 0, n_features - 1))
#     loss = -1 / n_samples * sp.Sum(loss_i ** 2, (i, 0, n_samples - 1))
#
#     test_inst1.set_cost(loss)
#     test_inst1.implement_methods()
#
#     # Method 2
#     test_inst2 = user_cost(use_hessian=True)
#
#     loss = lambda X, w, y: -1 * np.mean((X @ w) ** 2, axis=0)
#     grad = lambda X, w, y: -2 * np.mean(y * X, axis=0, keepdims=True).T
#     hess = lambda X, w, y: -2 * np.cov(X, rowvar=False)
#
#     test_inst2.set_cost(loss)
#     test_inst2.set_gradient(grad)
#     test_inst2.set_hessian(hess)
#
#     # Method 3
#     test_inst3 = negentropy_cost(source_name = 'exp',
#                                 source_params = {"alpha": 1})
#
#     #
#     w_ = np.ones((n_features, 1))
#     X_ = np.random.randn(n_samples, n_features)
#     mu_ = np.mean(X_, axis=0, keepdims=True)
#     std_ = np.std(X_, axis=0, keepdims=True)
#
#     X_ = (X_ - mu_) / std_
#     y_ = X_ @ w_
#
#     # Check derivatives
#     for cnt, test_inst in enumerate([test_inst1, test_inst2, test_inst3]):
#         print(f"\n\nFor instance {cnt + 1}:")
#         check_grad = test_inst3.check_gradient(X_, w_, y_, 1e-4)
#         check_hess = test_inst3.check_hessian(X_, w_, y_, 1e-4)
