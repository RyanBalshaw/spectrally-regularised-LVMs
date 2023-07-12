# Copyright 2023-present Ryan Balshaw
"""
This set of methods that define general cost functions that
one can use, using user-defined analytical functions based off
SymPy and NumPy.

Additionally, there are two specific methods implemented here:
- principal component analysis
- negentropy-based independent component analysis.
"""
import warnings

import numpy as np
import sympy as sp

from .negen_approx import initialise_sources


class CostClass(object):
    """
    Base class for different formulations of the user cost function. All children
    classes are expected to have _cost, _cost_gradient, and _cost_hessian instance
    attribute. These attributes are accessed by the methods of CostClass
    for ease of use.

    Methods
    -------
    cost(X, w, y)
        This method accesses the internal self._cost instance attribute and
        returns its output self._cost(X, w, y).

    cost_gradient(X, w, y)
        This method accesses the internal self._cost_gradient instance attribute and
        returns its output self._cost_gradient(X, w, y).

    cost_hessian(X, w, y)
        This method accesses the internal self._cost_hessian instance attribute and
        returns its output self._cost_hessian(X, w, y).

    finite_difference_grad(X, w, y, step_size)
        The method returns the central finite difference approximation to the gradient.

    finite_difference_hess(X, w, y, step_size)
        The method returns the central finite difference approximation to the Hessian.

    check_gradient(X, w, y, step_size)
        This method takes in an initial set of variables X, w, y, and a
        finite difference step size. The function is used to check the
        self._cost_gradient method using a central finite-difference
        approach.

    check_hessian(X, w, y, step_size)
        This method takes in an initial set of variables X, w, y, and a
        finite difference step size. The function is used to check the
        self._cost_hessian method using a central finite-difference
        approach.
    """

    def __init__(
        self,
        use_hessian: bool = True,
        verbose: bool = False,
        finite_diff_flag: bool = False,
    ):
        """

        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the parameter estimation step, as you may wish to just
            perform steepest descent instead of using Newton's method.

        verbose : bool
            A boolean flag to control any possible
            print statements.

        finite_diff_flag: bool
            A boolean flag to control whether a finite difference approximation
            is to be used for the gradient and hessian
        """

        self.use_hessian = use_hessian
        self.verbose = verbose
        self.finite_diff_flag = finite_diff_flag

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
        cost function evaluation of (X, w, y)
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
        derivative function evaluation of (X, w, y)
        """

        if self.finite_diff_flag:
            return self.finite_difference_grad(X, w, y, step_size=1e-4)

        else:
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
        Hessian function evaluation of (X, w, y)
        """
        if self.use_hessian:
            if self.finite_diff_flag:
                return self.finite_difference_hess(X, w, y, step_size=1e-4)

            else:
                return self._cost_hessian(X, w, y)

        else:
            return np.eye(w.shape[0])

    def finite_difference_grad(self, X, w, y, step_size: float):
        """
        Finite difference gradient approximation (central difference)

        Parameters
        ----------
        X : ndarray
            An array of size n_samples x n_features.

        w : ndarray
            An column vector of size n_features x 1

        y : ndarray
            An column vector of size n_features x 1. Expected to be
            equivalent to X @ w.

        step_size : float (default = 1e-4)
            The finite difference step size.

        Returns
        -------
        grad_fd : ndarray
            The finite difference approximation to the gradient
        """

        if self.verbose:
            print("Finite difference gradient approximation...")

        grad_fd = np.zeros_like(w)
        w0 = w.copy()

        for i in range(grad_fd.shape[0]):
            e_i = np.zeros_like(w0)
            e_i[i, 0] = step_size

            # Calculate sensitivity change in y as well!
            y_f = X @ (w0 + e_i)
            y_b = X @ (w0 - e_i)

            f_f = self._cost(X, w0 + e_i, y_f)
            f_b = self._cost(X, w0 - e_i, y_b)

            grad_fd[i, 0] = (f_f - f_b) / (2 * step_size)

        return grad_fd

    def finite_difference_hess(self, X, w, y, step_size: float):
        """
        Finite difference Hessian approximation (central difference)

        Parameters
        ----------
        X : ndarray
            An array of size n_samples x n_features.

        w : ndarray
            An column vector of size n_features x 1

        y : ndarray
            An column vector of size n_features x 1. Expected to be
            equivalent to X @ w.

        step_size : float (default = 1e-4)
            The finite difference step size.

        Returns
        -------
        hess_fd : ndarray
            The finite difference approximation to the hessian
        """

        if self.verbose:
            print("Finite difference hessian approximation...")

        D, _ = w.shape

        w0 = w.copy()

        hess_fd = np.zeros((D, D))

        for i in range(D):
            e_i = np.zeros_like(w0)
            e_i[i, 0] = step_size

            for j in range(D):
                e_j = np.zeros_like(w0)
                e_j[j, 0] = step_size

                # Calculate sensitivity change in y as well!
                y1 = X @ (w0 + e_i + e_j)
                y2 = X @ (w0 + e_i - e_j)
                y3 = X @ (w0 - e_i + e_j)
                y4 = X @ (w0 - e_i - e_j)

                f1 = self._cost(X, w0 + e_i + e_j, y1)
                f2 = self._cost(X, w0 + e_i - e_j, y2)
                f3 = self._cost(X, w0 - e_i + e_j, y3)
                f4 = self._cost(X, w0 - e_i - e_j, y4)

                hess_fd[i, j] = (f1 - f2 - f3 + f4) / (4 * step_size**2)

        return hess_fd

    def check_gradient(self, X, w, y, step_size: float = 1e-4):
        """
        This method checks the self._cost_gradient function to determine
        whether the gradient implementation is correct based off the
        objective function.

        Parameters
        ----------
        X : ndarray
            An array of size n_samples x n_features.

        w : ndarray
            An column vector of size n_features x 1

        y : ndarray
            An column vector of size n_features x 1. Expected to be
            equivalent to X @ w.

        step_size : float (default = 1e-4)
            The finite difference step size.

        Returns
        -------
        grad_current : ndarray
            The gradient based off the internal self._cost_gradient instance.

        grad_fd : ndarray
            The finite-difference approximation to the gradient

        grad_norm : ndarray
            The L2 norm between the analytical gradient and the finite difference
            approximation.

        Note that this is a helper method. costClass operates as a base class,
        so you will find that methods such as self.cost() and self.cost_gradient()
        are accessed but never defined. I use a child class to define these methods.
        """

        if self.verbose:
            print("\nChecking the gradient using central difference approximation...")

        if self.finite_diff_flag:
            warnings.warn(
                "finite_diff_flag = True, evaluation of gradient norm" " will be zero.",
                RuntimeWarning,
            )

        w0 = w.copy().reshape(-1, 1)

        grad_current = self.cost_gradient(X, w0, y).reshape(-1, 1)
        grad_fd = self.finite_difference_grad(X, w0, y, step_size)

        grad_norm = np.linalg.norm(grad_current - grad_fd)

        if self.verbose:
            print(f"Finished! The gradient norm is: {grad_norm}")

        return grad_current, grad_fd, grad_norm

    def check_hessian(self, X, w, y, step_size: float = 1e-4):
        """
        This method checks the self._cost_hessian function to determine
        whether the hessian implementation is correct based off the
        user-defined objective function.

        Parameters
        ----------
        X : ndarray
            An array of size n_samples x n_features.

        w : ndarray
            An column vector of size n_features x 1

        y : ndarray
            An column vector of size n_features x 1. Expected to be
            equivalent to X @ w.

        step_size : float (default = 1e-4)
            The finite difference step size.

        Returns
        -------
        hess_current : ndarray
            The hessian based off the internal self._cost_hessian instance.

        hess_check : ndarray
            The finite-difference approximation to the hessian.

        hess_norm : ndarray
            The L2 norm (average of the row-wise L2 norm) between the analytical
            hessian and the finite difference approximation.

        Note that this is a helper method. costClass operates as a base class,
        so you will find that methods such as self.cost() and self.cost_hessian()
        are accessed but never defined. I use a child class to define these methods.
        """

        if self.verbose:
            print("\nChecking the hessian using central difference approximation...")

        if self.finite_diff_flag:
            warnings.warn(
                "finite_diff_flag = True, evaluation of Hessian norm" " will be zero.",
                RuntimeWarning,
            )

        w0 = w.copy().reshape(-1, 1)
        hess_current = self.cost_hessian(X, w0, y)
        hess_fd = self.finite_difference_hess(X, w0, y, step_size)

        hess_norm = np.linalg.norm(hess_current - hess_fd)

        if self.verbose:
            print(f"Finished! The hessian norm is: {hess_norm}")

        return hess_current, hess_fd, hess_norm


class ExplicitCost(CostClass):
    """
    An object that implements the general user cost function
    class. This allows the user to manually define their cost function
    and associated gradient vector and hessian. Inherits from costClass.

    The user is asked to define their objective function, gradient function, and
    hessian function which take in three inputs: X, w, y. This can be done using
    the set_ methods that are available to an instance of this class, which is
    inherited from costClass.

    Assumed function format from user: func(X, w, y) where X is a ndarray
    with shape (n_samples, n_features), w is a ndarray with shape (n_features, 1)
    and y is the linear transformation X @ w with shape (n_samples, 1).

    Methods
    -------
    set_cost(cost_func)
        This method takes in a cost_func variable and sets it as
        an internal attribute self._cost.

    set_gradient(cost_gradient)
        This method takes in a cost_gradient variable and sets it as
        an internal attribute self._cost_gradient.

    set_hessian(cost_hessian)
        This method takes in a cost_hessian variable and sets it as
        an internal attribute self._cost_hessian.

    get_cost()
        This method returns the internal self._cost attribute.

    get_gradient()
        This method returns the internal self._cost_gradient attribute.

    get_hessian()
        This method returns the internal self._cost_hessian attribute.
    """

    def __init__(
        self,
        use_hessian: bool = True,
        verbose: bool = False,
        finite_diff_flag: bool = False,
    ):
        """
        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the parameter estimation step, as you may wish to just
            perform steepest descent instead of using Newton's method.

        verbose : bool
            A boolean flag to control any possible
            print statements.

        finite_diff_flag: bool
            A boolean flag to control whether a finite difference approximation
            is to be used for the gradient and hessian
        """
        super().__init__(use_hessian, verbose, finite_diff_flag)

    def set_cost(self, cost_func):
        """
        This method allows one to set their cost function.

        Parameters
        ----------
        cost_func : function
            The users cost function.

        Examples
        --------
        cost_func = lambda X, w, y: -1 * np.mean(y ** 2, axis=0)

        """
        self._cost = cost_func

    def set_gradient(self, cost_gradient):
        """
        This method allows one to set their gradient vector.

        Parameters
        ----------
        cost_gradient : function
            The users gradient vector of the cost function.

        Examples
        --------
        cost_gradient = lambda X, w, y: -2 * np.mean(y * X, axis=0,
            keepdims=True)

        """
        self._cost_gradient = cost_gradient

    def set_hessian(self, cost_hessian):
        """
        This method allows one to set their objective Hessian (optional).

        Parameters
        ----------
        cost_hessian : function
            The users gradient vector of the cost function.

        Examples
        --------
        cost_hessian = lambda X, w, y: -2 /X.shape[0] * (X.T @ X)

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


class SymbolicCost(CostClass):
    """
    This class implements a general user cost function class based off SymPy.
    This allows the user to manually define a symbolic representation of their
    cost function, and the necessary higher-order derivatives are calculated
    symbolically.

    Inherits from costClass.

    The user is asked to define their objective function loss based off of three inputs:
    z, its indexable variable i, and the number of indices n, i.e. i in [0, n - 1]. In
    code, z[i] = w^T x_i represents the latent transform of the ith x vector.

    This can be done using the set_ methods that are available to an instance of this
    class, which is inherited from costClass.

    Methods
    -------
    set_cost(cost_func)
        This method takes in the symbolic expression of a users cost function and then
        stores it as a class instance.

    get_sympy_parameters()
        This method gets the indexed random variable z, its index variable i and n,
        where n is the number of samples z[i], i = 0, ..., n - 1. Example cost function:
        loss = -1/n * sp.Sum((z[i])**2, (i))

    implement_cost()
        This method converts the symbolic loss/cost function to a numerical form.

    implement_first_derivative()
        This method converts the symbolic derivative of the loss to a numerical form.

    implement_second_derivative()
        This method converts the symbolic second derivative (index-wise) to a numerical
        form.

    implement_methods()
        This method runs all three implement_* methods in succession.

    _cost(X, w, y)
        This method returns the cost function value based off the symbolic loss.

    _cost_gradient(X, w, y)
        This method returns the gradient of the cost function based off the symbolic
        loss.

    _cost_hessian(X, w, y)
        This method returns the Hessian of the cost function based off the symbolic
        loss.

    """

    def __init__(
        self,
        use_hessian: bool = False,
        verbose: bool = False,
        finite_diff_flag: bool = False,
    ):
        """

        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the parameter estimation step, as you may wish to just
            perform steepest descent instead of using Newton's method.

        verbose : bool
            A boolean flag to control any possible
            print statements.

        finite_diff_flag: bool
            A boolean flag to control whether a finite difference approximation
            is to be used for the gradient and hessian
        """
        super().__init__(use_hessian, verbose, finite_diff_flag)

        if self.use_hessian and not self.finite_diff_flag:
            warnings.warn(
                "Running the Sympy cost with an exact hessian calculation can"
                " cause significant slowdowns. You may be better off using a"
                " finite difference or quasi-Newton approximation scheme.",
                RuntimeWarning,
            )

    def set_cost(self, cost_func):
        """
        This method allows one to set their cost function (overwrites default).

        Parameters
        ----------
        cost_func : function
            The users cost function defined symbolically.

        """
        self._sympy_cost = cost_func

        # Create the first and second derivative methods
        self.implement_methods()

    def get_symbolic_parameters(self):
        """

        Returns
        -------
        z: sp.IndexedBase instance
            An indexable variable that represents the transformation of the i^th
            data vector z_i = w.T @ x_i

        i: sp.Idx instance
            An indexable SymPy matrix of size (n_features, 1)

        n: sp.Idx instance
            A set of index variables that can be used to iterate over
            the X and w sp.IndexedBase instances.

        """

        # Create the size variable N
        self.n = sp.symbols("n", integer=True)

        # Create the iteration variables i, j, k
        self.j = sp.Idx("j", self.n)
        self.k = sp.Idx("k", self.n)
        self.l = sp.Idx("l", self.n)

        # Create the latent variable z (to be indexed)
        self.z = sp.IndexedBase("z")

        return self.z, self.j, self.n

    def implement_cost(self):
        """
        A method that lambdifies the user's cost function.
        """
        print("Lambdifying the sympy cost function...")

        self._cost_ = sp.lambdify([self.z, self.n], self._sympy_cost)
        # Will overwrite the sympy variable

    def implement_first_derivative(self):
        """
        A method that symbolically computes the gradient of the user's cost function,
        and then lambdifies it so that it can be used.

        The call occurs by iterating over the indices of w in (0, n_features - 1),
        deriving each gradient index using SymPy's .diff() method and storing each
        gradient computation in a SymPy matrix.
        """
        if self.verbose:
            print("Deriving and lambdifying the sympy gradient function...")

        cost_gradient = sp.diff(self._sympy_cost, self.z[self.k])

        self._cost_gradient_ = sp.lambdify([self.z, self.k, self.n], cost_gradient)

        # self._first_derivative_sympy = sp.Matrix(
        #     [self._sympy_cost.diff(self.w[i, 0]) for i in range(self.n_features)]
        # )
        # self._cost_gradient = sp.lambdify(
        #     (self.X, self.w, self.y), self._first_derivative_sympy
        # )

    def implement_second_derivative(self):
        """
        A method that symbolically computes the Hessian of the user's cost function, and
        then lambdifies it so that it can be used.

        The call occurs by iterating over the indices of w in (0, n_features - 1),
        deriving the gradient vector w.r.t w[i, 0] and storing the Hessian computation
        in a SymPy matrix.
        """
        if self.verbose:
            print("Deriving and lambdifying the sympy Hessian function...")

        if self.use_hessian:
            cost_gradient = sp.diff(self._sympy_cost, self.z[self.k])
            cost_hessian = sp.diff(cost_gradient, self.z[self.l])

            self._cost_hessian_ = sp.lambdify(
                [self.z, self.k, self.l, self.n], cost_hessian
            )

            # self._second_derivative_sympy = sp.BlockMatrix(
            #     [
            #         self._first_derivative_sympy.diff(self.w[i, 0])
            #         for i in range(self.n_features)
            #     ]
            # )
            # self._cost_hessian = sp.lambdify(
            #     (self.X, self.w, self.y), self._second_derivative_sympy
            # )

        else:
            self._cost_hessian = lambda X, w, y: np.eye(w.shape[0])

    def implement_methods(self):
        """
        This method combines the implement_* methods into one call. The idea was to
        provide access to a set lambdification process through one instance call.

        Raises
        -------
        AttributeError
            This is raised if the user's cost function has not been defined
            within the instance.
        """
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
            raise AttributeError

    def _cost(self, X, w, y):
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
        cost function evaluation of (X, w, y)
        """

        if not hasattr(self, "_cost"):
            self.implement_methods()  # Ensure creation of _methods.

        N = X.shape[0]

        return self._cost_(y[:, 0], N)

    def _cost_gradient(self, X, w, y):
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
        derivative function evaluation of (X, w, y)
        """

        N, D = X.shape

        grad_vector = np.zeros((D, 1))

        for i in range(N):
            grad_vector += self._cost_gradient_(y[:, 0], i, N) * X[[i], :].T

        return grad_vector

    def _cost_hessian(self, X, w, y):
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
        Hessian function evaluation of (X, w, y)
        """

        N, D = X.shape

        hess_mat = np.zeros((D, D))

        for j in range(N):
            x_j = X[[j], :].T

            hess_inner = np.zeros((D, D))

            for k in range(N):
                x_k = X[[k], :].T

                hess_inner += self._cost_hessian_(y[:, 0], j, k, N) * (x_k @ x_j.T)

            hess_mat += hess_inner

        return hess_mat


class NegentropyCost(CostClass):
    """
    This class implements the Negentropy cost that is commonly applied to ICA
    via negentropy maximisation or kurtosis maximisation. Inherits from CostClass.

    Assumed function format: func(X, w, y) where X is a ndarray
    with shape (n_samples, n_features), w is a ndarray with shape (n_features, 1)
    and y is the linear transformation X @ w with shape (n_samples, 1).

    Methods
    -------
    _cost(X, w, y)
        This method returns the cost function value for the Negentropy loss.

    _cost_gradient(X, w, y)
        This method returns the gradient of the cost function for the Negentropy
        loss.

    _cost_hessian(X, w, y)
        This method returns the Hessian of the cost function for the Negentropy
        loss.
    """

    def __init__(
        self,
        source_name: str,
        source_params: dict,
        use_approx: bool = True,
        use_hessian: bool = True,
        verbose: bool = False,
        finite_diff_flag: bool = False,
    ):
        """

        Parameters
        ----------
        source_name : str
            The name of the source function used for negentropy estimation.
            Options: logcosh, exp, quad, cube

        source_params :  dict
            A dictionary containing the source parameter {'alpha': 1}

        use_approx : bool
            Defines whether a term in the Negentropy jacobian is approximated or not.
            This approximation is common in the ICA approach by hyvarinen.

        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the parameter estimation step, as you may wish to just
            perform steepest descent instead of using Newton's method.

        verbose : bool
            A boolean flag to control any possible
            print statements.

        finite_diff_flag: bool
            A boolean flag to control whether a finite difference approximation
            is to be used for the gradient and hessian
        """
        super().__init__(use_hessian, verbose, finite_diff_flag)

        self.source_name = source_name
        self.source_params = source_params  # dictionary of parameters
        self.use_approx = use_approx

        # Initialise the source PDFs
        self.source_instance, self.source_expectation = initialise_sources(
            source_name, self.source_params
        )

        # Enable compatability with CostClass
        # self.set_cost(self.cost)
        # self.set_gradient(self.cost_gradient)
        # self.set_hessian(self.cost_hessian)

    def _cost(self, X, w, y):  # Important to negentropy-based ICA
        """
        Negentropy-estimate calculation for the objective function.

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
        Negentropy cost in minimisation setting using (X, w, y)

        """

        if y.shape[1] == 1:
            EG_y = np.mean(self.source_instance.function(y))

        else:
            EG_y = np.mean(self.source_instance.function(y), axis=0)

        return -1 * (EG_y - self.source_expectation) ** 2

    def _cost_gradient(self, X, w, y):
        """
        A method that returns the negentropy cost function gradient for the inputs.

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
        Derivative of the negetropy function evaluation using (X, w, y)
        """

        g_y = self.source_instance.first_derivative(y)

        # Calculate the expectation
        expectation = np.mean(g_y * X, axis=0, keepdims=True).T

        # Calculate the derivative scale with the missing term
        r = np.mean(self.source_instance.function(y)) - self.source_expectation

        # Calculate the gradient vector
        grad_vector = -2 * r * expectation

        return grad_vector

    def _cost_hessian(self, X, w, y):
        """
        A method that returns the negentropy cost function Hessian for the inputs.

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
        Hessian of the negentropy function evaluation using (X, w, y)
        """
        N, m = X.shape

        # Compute g'(y)
        g_prime_y = self.source_instance.second_derivative(y)

        # Calculate the gradient vector
        negentropy_gradient = self.cost_gradient(X, w, y)

        # Calculate the scalar r term
        r = np.mean(self.source_instance.function(y)) - self.source_expectation

        # t0 = time.time()
        if self.use_approx:
            # t0 = time.time()
            expectation = np.mean(g_prime_y) * np.eye(m)
        # t1 = time.time()

        else:
            negentropy_gradient /= -2 * r
            expectation = np.zeros((m, m))

            for n in range(N):
                expectation += np.dot(X[[n], :].T, X[[n], :]) * g_prime_y[n]

            expectation /= N

        # Calculate the gradient outer product
        grad_outer = negentropy_gradient @ negentropy_gradient.T

        # Calculate the Jacobian (Hessian)
        jacobian = -2 * (grad_outer + r * expectation)

        return jacobian


class VarianceCost(ExplicitCost):
    """
    This method implements the PCA variance maximisation objective. It inherits from
    the UserCost class and simply implements the three necessary components.
    """

    def __init__(
        self,
        use_hessian: bool = True,
        verbose: bool = False,
        finite_diff_flag: bool = False,
    ):
        """
        Defines the attributes for the user_cost class

        Parameters
        ----------
        use_hessian : bool
            A flag to control whether you use the Hessian or not.
            This is useful in the parameter estimation step, as you may wish to just
            perform steepest descent instead of using Newton's method.

        verbose : bool
            A boolean flag to control any possible
            print statements.

        finite_diff_flag: bool
            A boolean flag to control whether a finite difference approximation
            is to be used for the gradient and hessian
        """
        super().__init__(use_hessian, verbose, finite_diff_flag)

        def loss(X, w, y):
            """
            The PCA variance maximisation cost function for an optimisation
            framework which performs minimisation.

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
            Negative of latent variance.
            """
            return -1 * np.mean((X @ w) ** 2, axis=0)

        def grad(X, w, y):
            """
            The gradient of the PCA variance maximisation cost function for an
            optimisation framework which performs minimisation.

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
            Gradient of the negative of latent variance.
            """
            return -2 * np.mean(y * X, axis=0, keepdims=True).T

        def hess(X, w, y):
            """
            The Hessian of the PCA variance maximisation cost function for an
            optimisation framework which performs minimisation.

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
            Hessian of the negative of latent variance.
            """
            N = X.shape[0]

            return -2 / N * (X.T @ X)

        self.set_cost(loss)
        self.set_gradient(grad)
        self.set_hessian(hess)


# if __name__ == "__main__":
# def l1_l2_norm(X, w, y):
# 	N = X.shape[0]
#
# 	E1 = np.mean(np.abs(y), axis=0)
# 	E2 = np.mean(y ** 2, axis=0)
#
# 	return -1 * np.sqrt(N) * E1 / np.sqrt(E2)
#
#
# def l1_l2_grad(X, w, y):
# 	N = X.shape[0]
#
# 	E1 = np.mean(np.abs(y), axis=0)
# 	E2 = np.mean(y ** 2, axis=0)
#
# 	p1 = np.mean(np.sign(y) * X, axis=0).T / np.sqrt(E2)
# 	p2 = -1 * E1 / ((E2) ** (3 / 2)) * np.mean(y * X, axis=0).T
#
# 	return -1 * np.sqrt(N) * (p1 + p2)
#
#
# def l1_l2_hessian(X, w, y):
# 	pass


# test_instance = user_cost(use_hessian=False)
# test_instance = NegentropyCost(source_name='exp', source_params={'alpha': 1})

# test_instance.set_cost(l1_l2_norm)
# test_instance.set_gradient(l1_l2_grad)
#
# Lw = 256
# X_ = np.random.randn(1000, Lw)
# w_ = np.random.randn(Lw, 1)
# y_ = X_ @ w_
#
# res_grad = test_instance.check_gradient(X_, w_, y_, 1e-4)
# res_hess = test_instance.check_gradient(X_, w_, y_, 1e-4)
