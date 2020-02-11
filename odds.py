from scipy.stats import poisson, skellam
from collections import OrderedDict

from optimizers import PoissonOptimizer, SkellamOptimizer

eps = 0.05 # value for check function minimum is found


def get_poisson_mu(n, p):
    """Computes poisson expected mean
    
    :param n: max value of X - Poisson random variable
    :type n: int
    :param p: probability of X <= n
    :type p: float
    :return: Poisson expected mean
    :rtype: float
    """
    poiss = PoissonOptimizer(n=n, p=p)
    # start_point = 0.1
    # while True:
    #     mu = poiss.optimize(start_point)
    #     if poiss.loss(mu) < poiss.loss(mu + eps):
    #         break
    #     else:
    #         start_point += 0.1
    mu = poiss.optimize()
    return mu


def get_skellam_mu(hdp_val, p_hdp, mu):
    sk_opt = SkellamOptimizer(k=hdp_val, a=p_hdp, mu=mu)
    mu1, mu2 = sk_opt.optimize(mu/2)
    # sk_opt.plot_loss()
    # start_point = 0.1
    # while True:
    #     mu1, mu2 = sk_opt.optimize(start_point)
    #     if sk_opt.loss(mu2) < sk_opt.loss(mu2 + eps):
    #         break
    #     else:
    #         start_point += 0.1
    return mu1, mu2


def odd_to_prob(odd):
    """Convert odds to chances
    
    :param odd: Odd value without margin
    :type odd: float
    :return: Probability 
    :rtype: float
    """
    return 1 / odd


def prob_to_odd(prob):
    """Convert chances to odds
    
    :param prob: Probability
    :type prob: float
    :return: Odd value without margin
    :rtype: float
    """
    return 1 / prob


def total_draw_no_bet(k, mu, d):
    """Total draw_no_bet
    
    :param k: Goal scored
    :type k: int
    :param mu: Poisson expected mean
    :type mu: float
    :param d: odds of half total
    :type d: float
    :return: total over, total under odds
    :rtype: float, float
    """
    p = poisson.pmf(k + 1, mu)
    tu = d * (1 - p)
    to = prob_to_odd(1 - odd_to_prob(tu))
    return to, tu


def handicap_draw_no_bet(k, mu1, mu2, d):
    """Handicap draw_no_bet
    
    :param k: Goal difference
    :type k: int
    :param mu1: Poisson expected mean for Home team
    :type mu1: float
    :param mu2: Poisson expected mean for Away team
    :type mu2: float
    :param d: odd of half handicap 
    :type d: float
    :return: two handicap odds
    :rtype: float, float
    """
    p = skellam.pmf(k - 1, mu1, mu2)
    hdp1 = d * (1 - p)
    hdp2 = prob_to_odd(1 - odd_to_prob(hdp1))
    return hdp1, hdp2


def compute_totals(mu, totals, cur_goals, prefix):
    """Compute draw_no_bet, half and quarter totals 
    
    :param mu: Poisson expected mean
    :type mu: float
    :param totals: total values, e.g. 0.5, 1.5 etc
    :type totals: list
    :param prefix: T for common total, T1 or T2 for individual
    :type prefix: string
    :param cur_goals: current goals scored
    :type cur_goals: int 
    :return: map 'total_odd_name': odd_value
    :rtype: dict
    """
    odds = dict()
    for total in totals:
        goals = int(total)
        total = cur_goals + total

        # prob less than total goals
        p = sum(poisson.pmf(g, mu) for g in range(0, goals+1))

        # 1/2 totals
        odds[f"{prefix} O {total}"] = round(prob_to_odd(1 - p), 5)
        odds[f"{prefix} U {total}"] = round(prob_to_odd(p), 5)

        # 1/4 asian totals
        if goals != 0:
            to = (1 - poisson.pmf(goals, mu)/2) / (1 - p)
            tu = (1 - poisson.pmf(goals, mu)/2) / (p - poisson.pmf(goals, mu) / 2)
            odds[f"{prefix} O {total - 0.25}"] = round(to, 5)
            odds[f"{prefix} U {total - 0.25}"] = round(tu, 5)
        
        # draw_no_bet totals
        to, tu = total_draw_no_bet(goals, mu, odds[f"{prefix} U {total}"])
        odds[f"{prefix} O {total + 0.5}"] = round(to, 5)
        odds[f"{prefix} U {total + 0.5}"] = round(tu, 5)

        # 3/4 asian totals
        to = (1 - poisson.pmf(goals + 1, mu) / 2) / (1 - p - poisson.pmf(goals + 1, mu) / 2)
        tu = (1 - poisson.pmf(goals + 1, mu) / 2) / p
        odds[f"{prefix} O {total + 0.25}"] = round(to, 5)
        odds[f"{prefix} U {total + 0.25}"] = round(tu, 5)

    return odds


def compute_handicaps(mu1, mu2, g1, g2, handicaps, pinnacle=True):
    """Compute draw_no_bet, half and quarter handicaps and 1X2, 12, 1X, X2
    
    :param mu1: Poisson expected mean for Home team
    :type mu1: float
    :param mu2: Poisson expected mean for Away team
    :type mu2: float
    :param g1: current home goals scored
    :type g1: int
    :param g2: current away goals scored
    :type g2: int
    :param handicaps: handicap values, e.g. -0.5, -1.5 and etc
    :type handicaps: list 
    :return: map 'handicap_odd_name': odd_value
    :rtype: dict
    """
    odds = dict()
    
    for i, handicap in enumerate(handicaps):
        hdp_val = int(handicap*(-1) + 1) if handicap < 0 else int(handicap*(-1))
        
        if pinnacle:
            handicap = handicap  # Pinnacle handicap
        else:
            handicap = handicap - (g1 - g2)  # standard handicap

        # prob more than handicap goals difference
        p = sum(skellam.pmf(hdp, mu1, mu2) for hdp in range(hdp_val, 20))

        # 1/2 handicap
        odds[f"HDP1 {handicap}"] = round(prob_to_odd(p), 5)
        odds[f"HDP2 {handicap*(-1)}"] = round(prob_to_odd(1 - p), 5)

        # 1/4 asian handicaps
        if i != 0:
            hdp1_q = (1 - skellam.pmf(hdp_val, mu1, mu2) / 2) / (p - skellam.pmf(hdp_val, mu1, mu2) / 2)
            hdp2_q = (1 - skellam.pmf(hdp_val, mu1, mu2) / 2) / (1 - p)
            odds[f"HDP1 {handicap - 0.25}"] = round(hdp1_q, 5)
            odds[f"HDP2 {handicap*(-1) + 0.25}"] = round(hdp2_q, 5)

        # draw_no_bet handicaps
        hdp1_int, hdp2_int = handicap_draw_no_bet(hdp_val, mu1, mu2, odds[f"HDP1 {handicap}"])
        odds[f"HDP1 {handicap + 0.5}"] = round(hdp1_int, 5)
        odds[f"HDP2 {handicap*(-1) - 0.5}"] = round(hdp2_int, 5)

        # 3/4 asian handicaps 
        hdp1_q = (1 - skellam.pmf(hdp_val - 1, mu1, mu2)/2) / p
        hdp2_q = (1 - skellam.pmf(hdp_val - 1, mu1, mu2)/2) / (1 - p - skellam.pmf(hdp_val - 1, mu1, mu2)/2)
        odds[f"HDP1 {handicap + 0.25}"] = round(hdp1_q, 5)
        odds[f"HDP2 {handicap*(-1) - 0.25}"] = round(hdp2_q, 5)
    
    # double chances and money line
    p_win1 = odd_to_prob(odds[f"HDP1 -0.5"])
    p_win2 = odd_to_prob(odds[f"HDP2 -0.5"]) 
    p_X = 1 - p_win1 - p_win2

    odds["ML 1"] = round(odds[f"HDP1 -0.5"], 5)
    odds["ML 2"] = round(odds[f"HDP2 -0.5"], 5)
    odds["ML X"] = round(prob_to_odd(p_X), 5)
    odds["DC 1X"] = round(prob_to_odd(p_win1 + p_X), 5)
    odds["DC X2"] = round(prob_to_odd(p_X + p_win2), 5)
    odds["DC 12"] = round(prob_to_odd(p_win1 + p_win2), 5)

    return odds


def correct_score(mu1, mu2, g1, g2):
    """Correct score table
    
    :param mu1: Poisson expected mean for Home team
    :type mu1: float
    :param mu2: Poisson expected mean for Away team
    :type mu2: float
    :param mu2: 
    :type mu2: int
    :param mu2: 
    :type mu2: int
    :return: map 'score': odd_value
    :rtype: dict
    """
    odds = dict()
    for cs_g1 in range(0, int(mu1) + 6):
        for cs_g2 in range(0, int(mu2) + 6):
            p1 = poisson.pmf(cs_g1, mu1)
            p2 = poisson.pmf(cs_g2, mu2)
            g1_sc = int(g1 + cs_g1)
            g2_sc = int(g2 + cs_g2)
            odds[f"CS {g1_sc}:{g2_sc}"] = round(prob_to_odd(p1*p2), 5)
    return odds


def both_teams_score(mu1, mu2, g1, g2):
    odds = {}
    p1 = 1 - poisson.pmf(0, mu1)
    p2 = 1 - poisson.pmf(0, mu2)

    if g1 == 0 and g2 == 0:
        odds["BTS YES"] = round(prob_to_odd(p1*p2), 5)
        odds["BTS NO"] = round(prob_to_odd(1 - p1*p2), 5)
    elif g1 == 0 and g2 != 0:
        odds["BTS YES"] = round(prob_to_odd(p1), 5)
        odds["BTS NO"] = round(prob_to_odd(1 - p1), 5)
    elif g1 != 0 and g2 == 0:
        odds["BTS YES"] = round(prob_to_odd(p2), 5)
        odds["BTS NO"] = round(prob_to_odd(1 - p2), 5)
    else:
        odds["BTS YES"] = 1
        odds["BTS NO"] = 1
    return odds


def fair_odd(odd, margin):
    return odd / (1 - margin)


def margin_2(odd_1, odd_2):
    return 1 - ((odd_1 * odd_2)/(odd_1 + odd_2))


# # margin from calculator
# def margin_2(odd_1, odd_2):
#     return 1/odd_1 + 1/odd_2 - 1
#
#
# def fair_odd(odd, margin):
#     return odd * (margin + 1)


def clear_odds(odd_1, odd_2):
    margin = margin_2(odd_1, odd_2)
    fair_odd_1 = fair_odd(odd_1, margin)
    fair_odd_2 = fair_odd(odd_2, margin)
    prob_1 = odd_to_prob(fair_odd_1)
    prob_2 = odd_to_prob(fair_odd_2)
    return fair_odd_1, fair_odd_2


def compute_odds(k1, TO, TU, k2, HDP1, HDP2, g1=0, g2=0, order=True):
    """[summary]
    
    :param k1: total .5
    :type k1: float
    :param TO: total over k1 odd
    :type TO: float
    :param TU: total under k1 odd
    :type TU: float
    :param k2: handicap +-.05
    :type k2: float
    :param HDP1: handicap team1 
    :type HDP1: float
    :param HDP2: handicap team2
    :type HDP2: float
    :param g1: team home goals score
    :type g1: int
    :param g2: team away goals score
    :type g2: int
    :return: map 'odd': odd_value
    :rtype: dict
    """

    assert k1 % 1 == 0.5 and k1 > 0
    assert abs(k2 % 1) == 0.5

    TO, TU = clear_odds(TO, TU)
    HDP1, HDP2 = clear_odds(HDP1, HDP2)

    odds = dict()
    totals = [i + 0.5 for i in range(0, int(k1) + 6) if k1 + i > 0]
    handicap_diff = 4 if abs(k2) + int(abs(k2)) - 4 < 0 else int(2*abs(k2))
    handicaps = [k2 + i for i in range(int(k2) - handicap_diff, int(k2) + handicap_diff + 1)]
    goals = int(k1)

    # prob less than k1 goals
    p_under = odd_to_prob(TU)

    # get mu = mu1 + mu2 before game ending
    mu = get_poisson_mu(goals, p_under)

    # compute mu1 and mu2
    hdp_val = int(k2*(-1) + 1) if k2 < 0 else int(k2*(-1))
    p_hdp = odd_to_prob(HDP1)
    mu1, mu2 = get_skellam_mu(hdp_val, p_hdp, mu)

    # calculate odds
    ind_totals_1 = [i + 0.5 for i in range(0, int(mu1) + 4)]
    ind_totals_2 = [i + 0.5 for i in range(0, int(mu2) + 4)]
    odds_total = compute_totals(mu, totals, g1 + g2, prefix="T")
    odds_total_1 = compute_totals(mu1, ind_totals_1, g1, prefix="T1")
    odds_total_2 = compute_totals(mu2, ind_totals_2, g2, prefix="T2")
    odds_handicap = compute_handicaps(mu1, mu2, g1, g2, handicaps)
    odds_score = correct_score(mu1, mu2, g1, g2)
    odds_bts = both_teams_score(mu1, mu2, g1, g2)

    odds = {**odds_total, 
            **odds_total_1, 
            **odds_total_2, 
            **odds_handicap,
            **odds_score,
            **odds_bts}
    
    if order:
        odds = order_odds(odds)

    return odds, mu, mu1, mu2


def order_odds(odds):
    ordered_odds = OrderedDict()
    # totals
    for market in ['T', 'T1', 'T2']:
        odds_values = [float(x.split(' ')[2]) for x, y in odds.items() if x.split(' ')[0] == market]
        odds_values = sorted(odds_values)
        for y in odds_values:
            total_over = f'{market} O {y}'
            total_under = f'{market} U {y}' 
            ordered_odds[total_over] = odds.get(total_over)
            ordered_odds[total_under] = odds.get(total_under)

    # handicaps
    odds_values = [float(x.split(' ')[1]) for x, y in odds.items() if x.split(' ')[0] == 'HDP1']
    odds_values = sorted(odds_values)
    for y in odds_values:
        hdp1 = f'HDP1 {y}'
        hdp2 = f'HDP2 {-1 * y}' if y != 0. else f'HDP2 {y}'
        ordered_odds[hdp1] = odds.get(hdp1)
        ordered_odds[hdp2] = odds.get(hdp2)

    # other
    for x, y in odds.items():
        if x.split(' ')[0] in ['ML', 'DC', 'BTS', 'CS']:
            ordered_odds[x] = y

    return dict(ordered_odds)
