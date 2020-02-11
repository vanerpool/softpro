import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import poisson
from math import factorial


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
        return mu**k/factorial(k)*np.exp(-mu)
    
    def sum_pmf(self, mu):
        """sum of Poisson pmf's from 0 to n
        
        :param mu: Poisson expected mean
        :type mu: float
        :return: Poisson pmf from 0 to n
        :rtype: float
        """
        return sum(self.pmf(k, mu) for k in range(0, self.n+1))
    
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


if __name__ == '__main__':
    # p, n
    params = [
        (0.3554, 1),
        (0.5931, 2),
        (0.6200, 2),
        (0.6274, 2),
        (0.5790, 2)
    ]

    for p, n in params:
        poiss_opt = PoissonOptimizer(n=n, p=p)
        mu = poiss_opt.optimize(1)
        #plot graffics
        #poiss_opt.plot_loss()
        #plot_poisson_distribution(mu)
        print(f"mu={mu}")
