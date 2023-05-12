"""
This set of methods that define general cost functions that
one can use, and the FastICA cost function.
"""
import numpy as np
import sympy as sp

from .negen_approx import initialise_sources


class user_cost(object):
    """
    An object that implements the a general user cost function
     class. This allows the user to manually define their cost function
     and associated gradient vector and hessian.

     Assumed format for methods: func(X, w, y) where X is an ndarray
      with shape (n_samples, n_features), w is an ndarray with shape (n_features, 1)
      and y is the linear transformation X @ w with shape (n_samples, 1).

    """

    def __init__(self, use_hessian: bool):
        """

        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the solver, as you may wish to just
            perform steepest descent instead of using Newton's method.
        """
        self.use_hessian = use_hessian

    def set_cost(self, cost_func):
        """
        This method allows one to set their cost function.

        Parameters
        ----------
        cost_func : function
            The users cost function.
            Example: cost_func = lambda X, w, y: -1 * np.mean(y ** 2, axis=0)

        Raises
        -------
        AssertionError
            Error is thrown when cost_func is not of type 'function'.
        """
        try:
            assert type(cost_func).__name__ == "function"

        except AssertionError:
            ("Please implement a cost function that accepts parameters " "(X, w, y).")
            raise AssertionError

        else:
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

        Raises
        -------
        AssertionError
            Error is thrown when cost_gradient is not of type 'function'.
        """
        try:
            assert type(cost_gradient).__name__ == "function"

        except AssertionError:
            ("Please implement a gradient function that accepts parameters (X, w, y).")
            raise AssertionError

        else:
            self._cost_gradient = cost_gradient

    def set_hessian(self, cost_hessian):
        """
        This method allows one to set their objective Hessian (optional).

        Parameters
        ----------
        cost_hessian : function
            The users gradient vector of the cost function.
            Example: cost_hessian = lambda X, w, y: -2 /X.shape[0] * (X.T @ X)

        Raises
        -------
        AssertionError
            Error is thrown when cost_hessian is not of type 'function'.
        """
        try:
            assert type(cost_hessian).__name__ == "function"

        except AssertionError:
            ("Please implement a hessian function that accepts parameters (X, w, y).")
            raise AssertionError

        else:
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


class sympy_cost(object):
    def __init__(self, n_samples, n_features, use_hessian=False):
        self.n_samples = n_samples
        self.n_features = n_features
        self.use_hessian = use_hessian

    def get_model_parameters(self):
        i, j = sp.symbols("i j", cls=sp.Idx)

        w = sp.IndexedBase("w", shape=(self.n_features, 1))
        X = sp.IndexedBase("X", shape=(self.n_samples, self.n_features))

        return X, w, (i, j)

    def set_cost_sympy(self, cost_func_sympy, X, w):
        self.X = X
        self.w = w
        self._cost_sympy = cost_func_sympy

    def get_cost_sympy(self):
        return self._cost_sympy

    def implement_cost(self):
        self._cost = sp.lambdify((self.X, self.w), self._cost_sympy)

    def implement_first_derivative(self):
        self._first_derivative_sympy = sp.Matrix(
            [self._cost_sympy.diff(self.w[i, 0]) for i in range(self.n_features)]
        )
        self._first_derivative = sp.lambdify(
            (self.X, self.w), self._first_derivative_sympy
        )

    def implement_second_derivative(self):
        if self.use_hessian:
            self._second_derivative_sympy = sp.BlockMatrix(
                [
                    self._first_derivative_sympy.diff(self.w[i, 0])
                    for i in range(self.n_features)
                ]
            )
            self._second_derivative = sp.lambdify(
                (self.X, self.w), self._second_derivative_sympy
            )

        else:
            self._second_derivative = lambda X, w: np.eye(w.shape[0])

    def implement_methods(self):
        if hasattr(self, "cost_sympy"):
            self.implement_cost()
            self.implement_first_derivative()

            if self.use_hessian:
                self.implement_second_derivative()

        else:
            print(
                "Please first initialise the sympy cost function "
                "using set_cost_sympy()."
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

    def cost(self, X, w, *args):
        self.check_X(X)
        self.check_w(w)

        return self._cost(X, w)

    def cost_gradient(self, X, w, *args):
        return self._first_derivative(X, w)

    def cost_hessian(self, X, w, *args):
        if self.use_hessian:
            return self._second_derivative(X, w)

        else:
            return np.eye(w.shape[0])


class negentropy_cost(object):
    def __init__(self, source_name: str, source_params: dict):
        self.source_name = source_name
        self.source_params = source_params  # dictionary of parameters

        # Initialise the source PDFs
        self.source_instance, self.source_expectation = initialise_sources(
            source_name, self.source_params
        )

    def cost(self, X, w, y):  # Important to FastICA
        # Negentropy-estimate calculation

        if y.shape[1] == 1:
            EG_y = np.mean(self.source_instance.function(y))

        else:
            EG_y = np.mean(self.source_instance.function(y), axis=0)

        return -1 * (EG_y - self.source_expectation) ** 2

    def cost_gradient(self, X, w, y):  # Important to FastICA
        # t0 = time.time()

        N, m = X.shape

        g_y = self.source_instance.first_derivative(y)

        # Calculate the expectation
        expectation = np.mean(g_y * X, axis=0, keepdims=True).T

        # Calculate the derivative scale with the missing term
        r = np.mean(self.source_instance.function(y)) - self.source_expectation

        # Calculate the gradient vector
        grad_vector = -2 * r * expectation

        return grad_vector

    def cost_hessian(self, X, w, y, approx_flag=True):  # Important to FastICA
        # t0 = time.time()
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


# if __name__ == "__main__":
#
#     # Method 1
#     n_samples = 10000
#     n_features = 3
#
#     sympy_inst = sympy_cost(n_samples, n_features)
#
#     X, w, iter_params = sympy_inst.get_model_parameters()
#     i, j = iter_params
#
#     loss_i = sp.Sum(w[j, 0] * X[i, j], (j, 0, n_features - 1))
#     loss = 1 / n_samples * sp.Sum(loss_i ** 2, (i, 0, n_samples - 1))
#
#     sympy_inst.set_cost_sympy(loss, X, w)
#
#     w_ = np.ones((n_features, 1))
#     X_ = np.random.randn(n_samples, n_features)
#     mu_ = np.mean(X_, axis=0, keepdims=True)
#     std_ = np.std(X_, axis=0, keepdims=True)
#
#     X_ = (X_ - mu_) / std_
#     y_ = X_ @ w_
#
#     print("\nResults for method 2")
#     print("--------------------\n")
#     print("Loss function:")
#     print(sympy_inst.cost(X_, w_, y_))
#
#     print("\nDerivative:")
#     print(sympy_inst.cost_gradient(X_, w_, y_))
#
#     print("\nHessian:")
#     print(sympy_inst.cost_hessian(X_, w_, y_))
#
#     # Method 2
#     user_inst = user_cost()
#
#     linear_model = lambda X, w: X @ w
#     loss = lambda X, w, y: np.mean(y ** 2, axis = 0)
#     grad = lambda X, w, y: np.mean(2 * y * X, axis=0, keepdims=True).T
#     hess = lambda X, w, y: 2 / X.shape[0] * X.T @ X
#
#     user_inst.set_cost(loss)
#     user_inst.set_gradient(grad)
#     user_inst.set_hessian(hess)
#
#     print("\nResults for method 2")
#     print("--------------------\n")
#     print("Loss function:")
#     print(user_inst.cost(X_, w_, y_))
#
#     print("\nDerivative:")
#     print(user_inst.cost_gradient(X_, w_, y_))
#
#     print("\nHessian:")
#     print(user_inst.cost_hessian(X_, w_, y_))
