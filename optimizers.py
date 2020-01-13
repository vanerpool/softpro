import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import poisson, skellam
from scipy.special import iv  # modified Bessel function of the first kind
from math import factorial

N = 20  # Number of goals for Skellam distribution


class PoissonOptimizer:
    """Class for finding mu parameter of Poisson distibution
    """

    def __init__(self, n, p):
        """
        :param n: max value of X - Poisson random variable
        :type n: int
        :param p: probability of X <= n
        :type p: float
        """
        self.n = n
        self.p = p

    def pmf(self, k, mu):
        """Probability mass function of Poisson distribution
        :param k: Poisson random variable value
        :type k: int
        :param mu: Poisson expected mean
        :type mu: float
        :return: k-th Poisson pmf
        :rtype: float
        """
        return mu ** k / factorial(k) * np.exp(-mu)

    def sum_pmf(self, mu):
        """sum of Poisson pmf's from 0 to n

        :param mu: Poisson expected mean
        :type mu: float
        :return: Poisson pmf from 0 to n
        :rtype: float
        """
        return sum(self.pmf(k, mu) for k in range(0, self.n + 1))

    def loss(self, mu):
        """Loss function (MSE) for minimization task

        :param mu: Poisson expected mean
        :type mu: float
        :return: MSE loss
        :rtype: float
        """
        return abs(self.p - self.sum_pmf(mu))
        # return (self.p - self.sum_pmf(mu))**2

    def optimize(self):
        """Optimization method

        :return: returns mu
        :rtype: float
        """
        result = optimize.minimize(self.loss, self.n, method='Nelder-Mead')
        if result.message != 'Optimization terminated successfully.':
            print('algorithm does not converge')
            print(result)
            raise StopIteration
        return result.x[0]

    def plot_loss(self):
        """This method plots loss function
        """
        x = np.linspace(0.01, 10, 50)
        y = self.loss(x)
        plt.plot(x, y)
        plt.xlabel('mu')
        plt.ylabel('MSE loss')
        plt.show()


class SkellamOptimizer:
    """Class for finding mu1 and mu2 parameters of Skellam distibution
    """

    def __init__(self, k, a, mu):
        """
        :param k: initial difference between n1 and n2,
                  where n1, n2 - independent Poisson random variables
        :type k: int
        :param a: probability of n1-n2 > k
        :type a: float
        :param mu: truth value of mu1 + mu2
        :type mu: float
        """
        self.k = k
        self.a = a
        self.mu = mu

    def pmf(self, k, mu2):
        """Probability mass function of Skellam distribution,
           where mu1 = mu - mu2
        :param k: difference between n1 and n2
        :type k: int
        :param mu2: n2 expected mean
        :type mu2: float
        :return: k-th Skellam pmf
        :rtype: float
        """
        return np.exp(-self.mu) * ((self.mu - mu2) / mu2) ** (k / 2) * iv(k, 2 * np.sqrt((self.mu - mu2) * mu2))

    def sum_pmf(self, mu2):
        """sum of Skellam pmf's from initial k to N

        :param mu2: n2 expected mean
        :type mu2: float
        :return: Skellam pmf from k to N
        :rtype: float
        """
        return sum(self.pmf(k, mu2) for k in range(self.k, N))

    def loss(self, mu2):
        """Loss function (MSE) for minimization task

        :param mu2: n2 expected mean
        :type mu2: float
        :return: MSE loss
        :rtype: float
        """
        # return (self.a - self.sum_pmf(mu2))**2
        return abs(self.a - self.sum_pmf(mu2))

    def optimize(self, initial_point):
        """Optimization method

        :return: returns mu1 and mu2
        :rtype: float, float
        """
        result = optimize.minimize(self.loss, initial_point, method='Nelder-Mead')
        mu2 = result.x[0]
        mu1 = self.mu - mu2
        return mu1, mu2

    def plot_loss(self):
        """This method plots loss function
        """
        x = np.linspace(0.01, self.mu - 0.01, 50)
        y = self.loss(x)
        plt.plot(x, y)
        plt.xlabel('mu2')
        plt.ylabel('MSE loss')
        plt.show()


def plot_poisson_distribution(mu):
    """This function plots a Poisson distribution

    :param mu: expected mean of Poisson random variable
    :type mu: float
    """
    x = np.arange(poisson.ppf(0.01, mu),
                  poisson.ppf(0.99, mu))
    plt.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
    plt.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('k')
    plt.ylabel('pmf')
    plt.title(f"Poisson({round(mu, 4)}) distribution")
    plt.show()


def plot_skellam_distribution(mu1, mu2):
    """This function plots a Skellam distribution

    :param mu1: expected mean of Poisson random variable
    :type mu1: float
    :param mu2: expected mean of Poisson random variable
    :type mu2: float
    """
    x = np.arange(skellam.ppf(0.01, mu1, mu2),
                  skellam.ppf(0.99, mu1, mu2))
    plt.plot(x, skellam.pmf(x, mu1, mu2), 'bo', ms=8, label='skellam pmf')
    plt.vlines(x, 0, skellam.pmf(x, mu1, mu2), colors='b', lw=5, alpha=0.5)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('k')
    plt.ylabel('pmf')
    plt.title(f"Skellam({round(mu1, 4)},{round(mu2, 4)}) distribution")
    plt.show()
