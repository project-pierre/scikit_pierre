import unittest
from math import sqrt

from ....scikit_pierre.measures import inner_product


class TestInnerProduct(unittest.TestCase):

    def test_inner_product(self):
        answer = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5, 0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        self.assertEqual(inner_product.inner_product(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                     q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_harmonic_mean(self):
        answer = 2 * sum([(0.389 * 0.35) / (0.389 + 0.35), (0.5 * 0.563) / (0.5 + 0.563),
                          (0.25 * 0.4) / (0.25 + 0.4), (0.625 * 0.5) / (0.625 + 0.5),
                          (0.0 * 0.0) / 0.00001, (0.0 * 0.0) / 0.00001,
                          (0.25 * 0.0) / (0.25 + 0.0)])
        self.assertEqual(inner_product.harmonic_mean(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                     q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_cosine(self):
        answer_num = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5, 0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denominator = sqrt(sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2])) * \
                             sqrt(sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.cosine(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                              q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_denominator)

    def test_kumar_hassebrook(self):
        answer_num = sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5, 0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denominator = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                              sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2,
                                   0.0 ** 2])) - answer_num
        self.assertEqual(inner_product.kumar_hassebrook(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                        q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_denominator)

    def test_jaccard(self):
        answer_num = sum([(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2, (0.625 - 0.5) ** 2,
                          (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        answer_denominator = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                              sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2])) - \
                             sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5, 0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        self.assertEqual(inner_product.jaccard(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                               q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_denominator)

    def test_dice_similarity(self):
        answer_num = 2 * sum([0.389 * 0.35, 0.5 * 0.563, 0.25 * 0.4, 0.625 * 0.5, 0.0 * 0.0, 0.0 * 0.0, 0.25 * 0.0])
        answer_denominator = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                              sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.dice_similarity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                       q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_denominator)

    def test_dice_divergence(self):
        answer_num = sum([(0.389 - 0.35) ** 2, (0.5 - 0.563) ** 2, (0.25 - 0.4) ** 2, (0.625 - 0.5) ** 2,
                          (0.0 - 0.0) ** 2, (0.0 - 0.0) ** 2, (0.25 - 0.0) ** 2])
        answer_denominator = (sum([0.389 ** 2, 0.5 ** 2, 0.25 ** 2, 0.625 ** 2, 0.0 ** 2, 0.0 ** 2, 0.25 ** 2]) +
                              sum([0.35 ** 2, 0.563 ** 2, 0.4 ** 2, 0.5 ** 2, 0.0 ** 2, 0.0 ** 2, 0.0 ** 2]))
        self.assertEqual(inner_product.dice_divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                       q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_denominator)
