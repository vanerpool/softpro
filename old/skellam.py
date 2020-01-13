import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import skellam 
from scipy.special import iv # modified Bessel function of the first kind


N = 20
eps = 1e-3


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
        return np.exp(-self.mu)*((self.mu-mu2)/mu2)**(k/2)*iv(k, 2*np.sqrt((self.mu-mu2)*mu2))
    
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


if __name__ == '__main__':
    # a, k, mu
    test_params = [
        (0.2913, 1, 2.15055),
        (0.4837, 1, 2.3115),
        (0.6475, 1, 2.21013),
        (0.6032, 1, 2.184),
        (0.1543, 1, 2.34933),
        (0.6003, 1, 2.37216),
        (0.6089, 1, 2.17658),
        (0.4664, 1, 2.30392),
        (0.5850, 1, 2.49538),
        (0.2244, 1, 2.32281),
        (0.5656, 1, 2.31525),
        (0.1735, 1, 2.35313),
        (0.5871, 1, 2.09123),
        (0.6739, 1, 2.39889),
        (0.5698, 1, 2.27379),
        (0.5402, 1, 2.30392)
    ]

    for a, k, mu in test_params:
        sk_opt = SkellamOptimizer(k=k, a=a, mu=mu)
        mu1, mu2 = sk_opt.optimize(0.2)
        #plot graffics
        #sk_opt.plot_loss()
        #plot_skellam_distribution(mu1, mu2)
        print(f"mu1={mu1}, mu2={mu2}")
