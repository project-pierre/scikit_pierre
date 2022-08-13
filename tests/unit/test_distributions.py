import unittest
from pierre.distributions.weighted_strategy import weighted_strategy, weighted_probability_strategy
import pandas as pd


class TestBaseDistribution(unittest.TestCase):
    def setUp(self):
        self.item_classes_set_1 = pd.DataFrame([[0.5, 0.5, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0],
                                                [0.25, 0, 0.25, 0.25, 0, 0, 0.25]],
                                               columns=["Adventure", "Comedy", "Crime", "Drama",
                                                        "Romance", "Sci-fi", "Western"],
                                               index=['compadecida', 'amor', 'sol'])
        self.user_pref_set_1 = pd.DataFrame([['u-01', 'compadecida', 5],
                                             ['u-01', 'sol', 4],
                                             ['u-01', 'amor', 4],
                                             ],
                                            columns=["USER_ID", "ITEM_ID", "TRANSACTION_VALUE"])


class TestWeightedProbabilityStrategy(TestBaseDistribution):
    def test_1(self):
        answer1 = [round(a, 3) for a in [0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]]
        distribution = weighted_strategy(self.user_pref_set_1, self.item_classes_set_1)
        dist_list = [round(a, 3) for a in distribution.iloc[0].tolist()]
        self.assertEqual(dist_list, answer1)


class TestWeightedStrategy(TestBaseDistribution):
    def test_1(self):
        answer1 = [round(a, 3) for a in [0.193103493, 0.248275848, 0.124137924, 0.31034481, 0.0, 0.0, 0.124137924]]
        distribution = weighted_probability_strategy(user_pref_set=self.user_pref_set_1, item_classes_set=self.item_classes_set_1)
        dist_list = [round(a, 3) for a in distribution.iloc[0].tolist()]
        self.assertEqual(dist_list, answer1)
