import unittest

from ....scikit_pierre.measures import intersection


class TestIntersection(unittest.TestCase):
    def test_intersection_similarity(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                          min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(intersection.intersection_similarity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                              q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_intersection_divergence(self):
        answer_num = sum(
            [abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4),
             abs(0.625 - 0.5), abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(intersection.intersection_divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                              q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         (1 / 2) * answer_num)

    def test_wave_hedges(self):
        answer_num = sum([abs(0.389 - 0.35) / max([0.389, 0.35]), abs(0.5 - 0.563) / max([0.5, 0.563]),
                          abs(0.25 - 0.4) / max([0.25, 0.4]), abs(0.625 - 0.5) / max([0.625, 0.5]),
                          abs(0.0 - 0.0) / 0.00001, abs(0.0 - 0.0) / 0.00001, abs(0.25 - 0.0) / max([0.25, 0.0])])
        self.assertEqual(intersection.wave_hedges(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                  q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_czekanowski_similarity(self):
        answer_deno = sum([abs(0.389 + 0.35), abs(0.5 + 0.563), abs(0.25 + 0.4), abs(0.625 + 0.5),
                           abs(0.0 + 0.0), abs(0.0 + 0.0), abs(0.25 + 0.0)])
        answer_num = 2 * sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                              min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(intersection.czekanowski_similarity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_czekanowski_divergence(self):
        answer_deno = sum([abs(0.389 + 0.35), abs(0.5 + 0.563), abs(0.25 + 0.4), abs(0.625 + 0.5),
                           abs(0.0 + 0.0), abs(0.0 + 0.0), abs(0.25 + 0.0)])
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(intersection.czekanowski_divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                             q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_motyka_similarity(self):
        answer_deno = sum([abs(0.389 + 0.35), abs(0.5 + 0.563), abs(0.25 + 0.4), abs(0.625 + 0.5),
                           abs(0.0 + 0.0), abs(0.0 + 0.0), abs(0.25 + 0.0)])
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                          min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(intersection.motyka_similarity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                        q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_motyka_divergence(self):
        answer_deno = sum([abs(0.389 + 0.35), abs(0.5 + 0.563), abs(0.25 + 0.4), abs(0.625 + 0.5),
                           abs(0.0 + 0.0), abs(0.0 + 0.0), abs(0.25 + 0.0)])
        answer_num = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]), max([0.625, 0.5]),
                          max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(intersection.motyka_divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                        q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_kulczynski_s(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                          min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                           abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        self.assertEqual(intersection.kulczynski_s(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                   q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_ruzicka(self):
        answer_num = sum([min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
                          min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        answer_deno = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]), max([0.625, 0.5]),
                           max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(intersection.ruzicka(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                              q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_tanimoto(self):
        answer_num = sum([max([0.389, 0.35]) - min([0.389, 0.35]), max([0.5, 0.563]) - min([0.5, 0.563]),
                          max([0.25, 0.4]) - min([0.25, 0.4]), max([0.625, 0.5]) - min([0.625, 0.5]),
                          max([0.0, 0.0]) - min([0.0, 0.0]), max([0.0, 0.0]) - min([0.0, 0.0]),
                          max([0.25, 0.0]) - min([0.25, 0.0])])
        answer_deno = sum([max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]), max([0.625, 0.5]),
                           max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(intersection.tanimoto(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                               q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)
