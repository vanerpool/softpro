from odds import compute_odds
import pprint

# initial odds line params
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 2.204, 1.835,	0.5, 1.652, 2.537
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 1.2470, 3.88, 1.5, 3.53, 1.298
# k1, to, tu, k2, hdp1, hdp2 = 3.5, 1.649, 2.25, 2.5, 2.03, 1.813
# k1, to, tu, k2, hdp1, hdp2 = 3.5, 1.649, 2.25, 3.5, 1.432, 2.84
# k1, to, tu, k2, hdp1, hdp2 = 3.5, 1.649, 2.25, -3.5, 2.84, 1.432
# k1, to, tu, k2, hdp1, hdp2 = 3.5, 1.649, 2.25, -2.5, 1.813, 2.03
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 1.2470, 3.88, -1.5, 1.298, 3.53
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 2.36, 1.757,  0.5, 2.12, 1.714
# k1, to, tu, k2, hdp1, hdp2 = 1.5, 1.129, 4.78, 1.5, 4.86, 1.131
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 2.52, 1.5, -0.5, 1.578, 2.33
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 1.684, 2.14, -0.5, 1.578, 2.33
# k1, to, tu, k2, hdp1, hdp2 = 2.5, 2.43, 1.558, 0.5, 2.29, 1.641
k1, to, tu, k2, hdp1, hdp2 = 3.5, 2.38, 1.617, -0.5, 1.68, 2.29

if __name__ == "__main__":
    odds, _, _, _ = compute_odds(k1, to, tu, k2, hdp1, hdp2)

    pprint.pprint(odds)
