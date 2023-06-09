# Copyright 2023-present Ryan Balshaw
"""
The negentropy approximation functions for the negentropy-based ICA methods.
"""
import numpy as np


class LogcoshObject(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the logcosh function. These functions are used in the
    negentropy approximation calculation for negentropy-based ICA.

    Methods
    -------
    function(u = float) -> 1/a1 * log((exp(a1 * u) + exp(-a1 * u)) / 2)
            Return the function form of G(u) for the logcosh function

    first_derivative(u = float) -> tanh
            Return the function form of the first derivative g(u) for the logcosh
            function

    second_derivative(u = float) -> 1 - tanh^2(u)
            Return the function form of the second derivative g'(u) for the logcosh
            function

    gamma(u = float)
            Return the function form of ratio g'(u)/g(u) for the logcosh function
    """

    def __init__(self, a1=None):
        """

        Parameters
        ----------
        a1: float
                The a1 parameter for the  generalised logcosh function.

        Raises
        ------
        ValueError
                If the a1 value is outside the recommended domain of [1, 2]
        """
        if a1 is None:
            self.a1 = [1]  # [a1 = 1]

        else:
            if a1 < 1 or a1 > 2:
                print(
                    f"a1 parameter ({a1}) in logcosh object is outside [1, 2] bounds."
                )
                raise ValueError

            self.a1 = a1  # [a1]

    def function(self, u):
        """
        This method implements the functional form of G(u)

        Parameters
        ----------
        u: float
                The input value to be fed through G(u)

        Returns
        -------
        The computation of G(u)

        """

        return (
            1
            / self.a1
            * np.log((np.exp(self.a1 * u) + np.exp(-self.a1 * u)) / 2 + 1e-12)
        )  # 1e-12 for stability

    def first_derivative(self, u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g(u)

        Returns
        -------
        The computation of g(u)
        """
        # au = self.a1 * u
        # e_au = np.exp(au)
        # e_n_au = np.exp(-au)

        return np.tanh(self.a1 * u)
        # (e_au - e_n_au) / (
        # e_au + e_n_au)

    def second_derivative(self, u):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g'(u)

        Returns
        -------
        The computation of g'(u)
        """
        d1 = self.first_derivative(self.a1 * u)

        return 1 - self.a1 * d1 * d1

    def gamma(self, u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamma(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float):
                The input value to be fed through gamma(u)

        Returns
        -------
        The computation of gamma(u)
        """
        return self.second_derivative(u) / self.first_derivative(u)

    # I played around with finding the inverse distribution from the
    # transformed distribution.
    # Not useful here.

    # def inverse(self, p):
    #     # p represents the variables mapped to X -> P, we wish to recover X
    #
    #     e_p = np.exp(p)
    #
    #     return np.array([-1, 1]) * np.log(e_p + np.sqrt(e_p**2 - 1)).reshape(
    #         -1, 1
    #     )  # Can have two solutions
    #
    # def inverse_derivative(self, p):
    #
    #     e_p = np.exp(p)
    #     e_2p = e_p**2
    #
    #     numerator = e_p + e_2p * (e_2p - 1) ** (-1 / 2)
    #     denominator = e_p + np.sqrt(e_2p - 1)
    #     deriv = numerator / denominator
    #
    #     return np.array([-1, 1]) * deriv.reshape(-1, 1)


class ExpObject(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the exp function. These functions are used in the
    negentropy approximation calculation for negentropy-based ICA.

    Methods
    -------
    function(u = float) -> -1/a2 * exp(-a2/2 * u^2)
            Return the function form of G(u) for the exp function

    first_derivative(u = float) -> u * exp(-a2/2 * u^2)
            Return the function form of the first derivative g(u) for the exp function

    second_derivative(u = float) -> (1 - a2 * u^*2) * exp(-a2 / 2 * u^2)
            Return the function form of the second derivative g'(u) for the exp function

    gamma(u = float) -> (1 - a2 * u^2) / u
            Return the function form of ratio g'(u)/g(u) for the exp function
    """

    def __init__(self, a2=None):
        """
        Parameters
        ----------
        a2: float - default: 1
                The a2 parameter for the exp function.

        Raises
        ------
        ValueError
                If the a2 value is outside the recommended domain of [0, 2]
        """
        if a2 is None:
            self.a2 = [1]  # [a2 = 1]

        else:
            if a2 < 0 or a2 > 2:
                print(
                    f"a2 parameter ({a2}) in exp object "
                    f"is outside [0, 2] bounds (a2 should be 1)."
                )
                raise ValueError

            self.a2 = a2  # [a2]

    def function(self, u):
        """
        This method implements the functional form of G(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through G(u)

        Returns
        -------
        The computation of G(u)
        """
        return -1 / self.a2 * np.exp(-self.a2 / 2 * u**2)

    def first_derivative(self, u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g(u)

        Returns
        -------
        The computation of g(u)
        """
        return u * np.exp(-self.a2 * u * u / 2)

    def second_derivative(self, u):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g'(u)

        Returns
        -------
        The computation of g'(u)
        """
        return np.exp(-self.a2 * u * u / 2) * (1 - self.a2 * u * u)

    def gamma(self, u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamma(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float):
                The input value to be fed through gamma(u)

        Returns
        -------
        The computation of gamma(u)
        """
        return (1 - self.a2 * u * u) / u


class QuadObject(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the quad (u**4) function. These functions are used in the
    negentropy approximation calculation for negentropy-based ICA.

    Methods
    -------
    function(u = float) -> 1/4 u^4
            Return the function form of G(u) for the quartic function

    first_derivative(u = float) -> u^3
            Return the function form of the first derivative g(u) for the quartic
            function

    second_derivative(u = float) -> 3 u ^2
            Return the function form of the second derivative g'(u) for the quartic
            function

    gamma(u = float) -> 3 / u
            Return the function form of ratio g'(u)/g(u) for the quartic function
    """

    @staticmethod
    def function(u):
        """
        This method implements the functional form of G(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through G(u)

        Returns
        -------
        The computation of G(u)
        """
        return 1 / 4 * u**4

    @staticmethod
    def first_derivative(u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g(u)

        Returns
        -------
        The computation of g(u)
        """
        return u**3

    @staticmethod
    def second_derivative(u):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g'(u)

        Returns
        -------
        The computation of g'(u)
        """
        return 3 * u**2

    @staticmethod
    def gamma(u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamma(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float):
                The input value to be fed through gamma(u)

        Returns
        -------
        The computation of gamma(u)
        """
        return 3 / u


class CubeObject(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the quad (u**4) function. These functions are used in the
    negentropy approximation calculation for negentropy-based ICA.

    Methods
    -------
    function(u = float) -> u^3
            Return the function form of G(u) for the cube function

    first_derivative(u = float) -> 3 u^2
            Return the function form of the first derivative g(u) for the cube
            function

    second_derivative(u = float) -> 6 u
            Return the function form of the second derivative g'(u) for the cube
            function

    gamma(u = float) -> 2 / u
            Return the function form of ratio g'(u)/g(u) for the cube function
    """

    @staticmethod
    def function(u):
        """
        This method implements the functional form of G(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through G(u)

        Returns
        -------
        The computation of G(u)
        """
        return u**3

    @staticmethod
    def first_derivative(u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g(u)

        Returns
        -------
        The computation of g(u)
        """
        return 3 * u**2

    @staticmethod
    def second_derivative(u):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float):
                The input value to be fed through g'(u)

        Returns
        -------
        The computation of g'(u)
        """
        return 6 * u

    @staticmethod
    def gamma(u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamma(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float):
                The input value to be fed through gamma(u)

        Returns
        -------
                The computation of gamma(u)
        """
        return 2 / u


def initialise_sources(
    source_name: str = "logcosh",
    source_params: dict | None = None,
):
    """
    A function that takes in the source name and its associated parameters
    and returns the source instance and the E{G(nu)} value
    for a set number of samples (100 000 samples).

    Parameters
    ----------
    source_name: str
            The name of the source that is to be used.

    source_params: dict | None (default None)
            The dictionary of parameters for the associated approximator.
            Format: {"alpha": alpha_val} where alpha_val is some float.

    Returns
    -------
    source_instance:
            The source instance that is to be used.

    source_expecation: float
            The evaluation of G(nu) for a set number of samples.

    """
    if source_params is None:
        source_params = {"alpha": 1}

    if source_name.lower() == "logcosh":
        source_instance = LogcoshObject(source_params["alpha"])

    elif source_name.lower() == "exp":
        source_instance = ExpObject(source_params["alpha"])

    elif source_name.lower() == "quad":
        source_instance = QuadObject()

    elif source_name.lower() == "cube":
        source_instance = CubeObject()

    else:
        print("Source name ({}) is unknown. Exiting the function.".format(source_name))
        raise SystemExit

    # Initialise expected value for G(nu) (for the negentropy-based ICA objective)

    source_expectation = np.mean(
        source_instance.function(np.random.randn(100000))
    )  # 1 hundred thousand samples... hot damn

    return source_instance, source_expectation
