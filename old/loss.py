from scipy.stats import poisson, skellam
from scipy.optimize import minimize


class PoissonLosses:

    def __init__(self, p_under_new, goals):
        self.p_under_new = p_under_new
        self.goals = goals
    
    def loss_total_0(self, mu):
        sum_p = sum(poisson.pmf(g, mu) for g in range(0, self.goals))
        return (self.p_under_new - (sum_p / (1 - poisson.pmf(self.goals, mu))))**2

    def loss_total_0_25(self, mu):
        sum_p = sum(poisson.pmf(g, mu) for g in range(0, self.goals))
        return (self.p_under_new - ((poisson.pmf(self.goals, mu) / 2 + sum_p) / (1 - poisson.pmf(self.goals, mu) / 2)))**2

    def loss_total_0_5(self, mu):
        sum_p = sum(poisson.pmf(g, mu) for g in range(0, self.goals))
        return (self.p_under_new - sum_p)**2

    def loss_total_0_75(self, mu):
        sum_p = sum(poisson.pmf(g, mu) for g in range(0, self.goals))
        return (self.p_under_new - (sum_p / (1 - poisson.pmf(self.goals, mu) / 2)))**2


class SkellamHomeLosses:
    
    def __init__(self, p_hdp_new, hdp_val, mu):
        self.p_hdp_new = p_hdp_new
        self.hdp_val = hdp_val
        self.mu = mu
    
    def loss_hdp_0(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20))
        return (self.p_hdp_new - (sum_p / (1 - skellam.pmf(self.hdp_val - 1, self.mu - mu2, mu2))))**2

    def loss_hdp_0_25(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20))
        return (self.p_hdp_new - ((sum_p - skellam.pmf(self.hdp_val, self.mu - mu2, mu2)/2) / (1 - skellam.pmf(self.hdp_val, self.mu - mu2, mu2)/2)))**2

    def loss_hdp_0_5(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20))
        return (self.p_hdp_new - sum_p)**2

    def loss_hdp_0_75(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20))
        return (self.p_hdp_new - (sum_p / (1 - skellam.pmf(self.hdp_val-1, self.mu - mu2, mu2) / 2)))**2


class SkellamAwayLosses:
    
    def __init__(self, p_hdp_new, hdp_val, mu):
        self.p_hdp_new = p_hdp_new
        self.hdp_val = hdp_val
        self.mu = mu
    
    def loss_hdp_0(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20)) #from home team
        return (self.p_hdp_new - (1 - sum_p / (1 - skellam.pmf(self.hdp_val - 1, self.mu - mu2, mu2))))**2

    def loss_hdp_0_25(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20)) #from home team
        # print((1 - sum_p) / (1 - skellam.pmf(self.hdp_val, self.mu - mu2, mu2)/2))
        return (self.p_hdp_new - ((1 - sum_p) / (1 - skellam.pmf(self.hdp_val, self.mu - mu2, mu2)/2)))**2

    def loss_hdp_0_5(self, mu2):
        sum_p = sum(skellam.pmf(hdp, mu2, self.mu - mu2) for hdp in range(self.hdp_val, 20))
        return (self.p_hdp_new - sum_p)**2

    def loss_hdp_0_75(self, mu2):
        sum_p = sum(skellam.pmf(hdp, self.mu - mu2, mu2) for hdp in range(self.hdp_val, 20)) #from home team
        # print((1 - sum_p - skellam.pmf(self.hdp_val-1, self.mu - mu2, mu2)/2) / (1 - skellam.pmf(self.hdp_val-1, self.mu - mu2, mu2) / 2))
        return (self.p_hdp_new - ((1 - sum_p - skellam.pmf(self.hdp_val-1, self.mu - mu2, mu2)/2) / (1 - skellam.pmf(self.hdp_val-1, self.mu - mu2, mu2) / 2)))**2
