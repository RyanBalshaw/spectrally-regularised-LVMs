"""
The negentropy approximation functions for the FastICA methods.
"""
import numpy as np


class general_object(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the XXXX (fill in here) function. These functions are used in the
    negentropy calculation for FastICA.

    Methods
    -------
    function(u = float)
        Return the function form of G(u) for the XXXX function

    first_derivative(u = float)
        Return the function form of the first derivative g(u) for the XXXX function

    second_derivative(u = float)
        Return the function form of the second derivativeg'(u) for the XXXX function

    gamma(u = float)
        Return the function form of ratio g'(u)/g(u) for the XXXX function
    """

    def function(self, u: float):
        """
        This method implements the functional form of G(u)

        Parameters
        ----------
        u: (float): The input value to be fed through G(u)

        Returns
        -------
            float: The computation of G(u)
        """

        return None

    def first_derivative(self, u: float):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float): The input value to be fed through g(u)

        Returns
        -------
            float: The computation of g(u)
        """

        return None

    def second_derivative(self, u: float):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float): The input value to be fed through g'(u)

        Returns
        -------
            float: The computation of g'(u)
        """

        return None

    def gamma(self, u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamm(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float): The input value to be fed through gamma(u)

        Returns
        -------
            float: The computation of gamma(u)
        """

        return None


class logcosh_object(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the logcosh function. These functions are used in the
    negentropy approximation calculation for FastICA.

    Methods
    -------
    function(u = float)
        Return the function form of G(u) for the logcosh function

    first_derivative(u = float)
        Return the function form of the first derivative g(u) for the logcosh function

    second_derivative(u = float)
        Return the function form of the second derivativeg'(u) for the logcosh function

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
        u: (float): The input value to be fed through G(u)

        Returns
        -------
            float: The computation of G(u)
        """

        return (
            1
            / (self.a1)
            * np.log((np.exp(self.a1 * u) + np.exp(-self.a1 * u)) / 2 + 1e-12)
        )  # 1e-12 for stability

    def first_derivative(self, u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float): The input value to be fed through g(u)

        Returns
        -------
            float: The computation of g(u)
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
        u: (float): The input value to be fed through g'(u)

        Returns
        -------
            float: The computation of g'(u)
        """
        d1 = self.first_derivative(self.a1 * u)

        return 1 - self.a1 * d1 * d1

    def gamma(self, u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamm(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float): The input value to be fed through gamma(u)

        Returns
        -------
            float: The computation of gamma(u)
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


class exp_object(object):
    """
    An object that implements the first derivative, second derivative and
    gamma functions of the exp function. These functions are used in the
    negentropy approximation calculation for FastICA.

    Methods
    -------
    function(u = float)
        Return the function form of G(u) for the exp function

    first_derivative(u = float)
        Return the function form of the first derivative g(u) for the exp function

    second_derivative(u = float)
        Return the function form of the second derivativeg'(u) for the exp function

    gamma(u = float)
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
        u: (float): The input value to be fed through G(u)

        Returns
        -------
            float: The computation of G(u)
        """
        return -1 / self.a2 * np.exp(-self.a2 / 2 * u**2)

    def first_derivative(self, u):
        """
        This method implements the first derivative of G(.) for g(u)

        Parameters
        ----------
        u: (float): The input value to be fed through g(u)

        Returns
        -------
            float: The computation of g(u)
        """
        return u * np.exp(-self.a2 * u * u / 2)

    def second_derivative(self, u):
        """
        This method implements the second derivative of G(.) for g'(u)

        Parameters
        ----------
        u: (float): The input value to be fed through g'(u)

        Returns
        -------
            float: The computation of g'(u)
        """
        return np.exp(-self.a2 * u * u / 2) * (1 - self.a2 * u * u)

    def gamma(self, u):
        """
        This method implements the ratio of the second derivative to
        the first derivative (gamm(u) = g'(u) / g(u))

        Parameters
        ----------
        u: (float): The input value to be fed through gamma(u)

        Returns
        -------
            float: The computation of gamma(u)
        """
        return (1 - self.a2 * u * u) / u
