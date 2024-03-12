import unittest
from ...scikit_pierre.tradeoff_weight.weight import genre_count, norm_var


class TestBaseWeights(unittest.TestCase):
    def setUp(self):
        self.test1 = [0.5, 0.2, 0.1, 0.0, 0.0, 0.0]
        self.test2 = [0.269, 0.192, 0.076, 0.384, 0.0, 0.0, 0.076]
        self.test3 = [0.1, 0.8, 0.0, 0.5, 0.6]
        self.test4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.test5 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.test6 = [0.388, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25]


class TestWeightGenreCount(TestBaseWeights):
    def test_1(self):
        answer1 = 3/6
        self.assertEqual(genre_count(self.test1), answer1)

    def test_2(self):
        answer2 = 5/7
        self.assertEqual(genre_count(self.test2), answer2)

    def test_3(self):
        answer3 = 4/5
        self.assertEqual(genre_count(self.test3), answer3)

    def test_4(self):
        answer4 = 0/6
        self.assertEqual(genre_count(self.test4), answer4)

    def test_5(self):
        answer5 = 6/6
        self.assertEqual(genre_count(self.test5), answer5)

    def test_6(self):
        answer6 = 5/7
        self.assertEqual(genre_count(self.test6), answer6)


class TestWeightVar(TestBaseWeights):
    def test_1(self):
        answermean = sum(self.test1)/len(self.test1)
        answer1 = 1 - sum(abs(j - answermean) ** 2 for j in self.test1)/len(self.test1)
        self.assertEqual(norm_var(self.test1), answer1)

    def test_2(self):
        answermean = sum(self.test2)/len(self.test2)
        answer2 = 1 - sum(abs(j - answermean) ** 2 for j in self.test2)/len(self.test2)
        self.assertEqual(norm_var(self.test2), answer2)

    def test_3(self):
        answermean = sum(self.test3)/len(self.test3)
        answer3 = 1 - (sum(abs(j - answermean) ** 2 for j in self.test3)/len(self.test3))
        self.assertEqual(norm_var(self.test3), answer3)

    def test_4(self):
        answermean = sum(self.test4)/len(self.test4)
        answer4 = 1 - sum(abs(j - answermean) ** 2 for j in self.test4)/len(self.test4)
        self.assertEqual(norm_var(self.test4), answer4)

    def test_5(self):
        answermean = sum(self.test5)/len(self.test5)
        answer5 = 1 - sum(abs(j - answermean) ** 2 for j in self.test5)/len(self.test5)
        self.assertEqual(norm_var(self.test5), answer5)

    def test_6(self):
        answermean = sum(self.test6)/len(self.test6)
        answer6 = 1 - sum(abs(j - answermean) ** 2 for j in self.test6)/len(self.test6)
        self.assertEqual(norm_var(self.test6), answer6)


if __name__ == '__main__':
    unittest.main()
