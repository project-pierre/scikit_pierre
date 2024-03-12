import numpy as np
import unittest
from copy import deepcopy

from ...scikit_pierre.relevance.relevance_measures import sum_relevance_score, ndcg_relevance_score


class TestBaseRelevance(unittest.TestCase):
    def setUp(self):
        self.test1 = [4, 5, 4.5, 4, 5]
        self.test2 = [5, 5, 5, 5, 5, 5]
        self.test3 = [100, 80, 90, 50, 90]
        self.test4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.test5 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.test6 = [0.388, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]


class TestSumRelevance(TestBaseRelevance):
    def test_1(self):
        answer1 = sum(self.test1)
        self.assertEqual(sum_relevance_score(self.test1), answer1)

    def test_2(self):
        answer2 = sum(self.test2)
        self.assertEqual(sum_relevance_score(self.test2), answer2)

    def test_3(self):
        answer3 = sum(self.test3)
        self.assertEqual(sum_relevance_score(self.test3), answer3)

    def test_4(self):
        answer4 = sum(self.test4)
        self.assertEqual(sum_relevance_score(self.test4), answer4)

    def test_5(self):
        answer5 = sum(self.test5)
        self.assertEqual(sum_relevance_score(self.test5), answer5)

    def test_6(self):
        answer6 = sum(self.test6)
        self.assertEqual(sum_relevance_score(self.test6), answer6)


class TestNDCGRelevance(TestBaseRelevance):
    def test_1(self):
        l = deepcopy(self.test1)
        dcg = sum(((2 ** w) - 1) / (np.log2(i + 2)) for i, w in enumerate(l))

        l.sort(reverse=True)
        idcg = sum(((2 ** w) - 1) / (np.log2(i + 2)) for i, w in enumerate(l))

        answer1 = dcg / idcg
        self.assertEqual(ndcg_relevance_score(self.test1), answer1)

    def test_2(self):
        l = deepcopy(self.test2)
        dcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        l.sort(reverse=True)
        idcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        answer2 = dcg / idcg
        self.assertEqual(ndcg_relevance_score(self.test2), answer2)

    def test_3(self):
        l = deepcopy(self.test3)
        dcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        l.sort(reverse=True)
        idcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        answer3 = dcg / idcg
        self.assertEqual(ndcg_relevance_score(self.test3), answer3)

    def test_4(self):
        self.assertEqual(ndcg_relevance_score(self.test4), 0.0)

    def test_5(self):
        l = deepcopy(self.test5)
        dcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        l.sort(reverse=True)
        idcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        answer5 = dcg / idcg
        self.assertEqual(ndcg_relevance_score(self.test5), answer5)

    def test_6(self):
        l = deepcopy(self.test6)
        dcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        l.sort(reverse=True)
        idcg = sum(((2 ** w) - 1) / (np.log2((i + 1) + 1)) for i, w in enumerate(l))

        answer6 = dcg / idcg
        self.assertEqual(ndcg_relevance_score(self.test6), answer6)


if __name__ == '__main__':
    unittest.main()
