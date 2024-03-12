import unittest
from math import sqrt, log

from ....scikit_pierre.measures import fidelity


class TestFidelity(unittest.TestCase):

    def test_fidelity(self):
        answer = sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4), sqrt(0.625 * 0.5), sqrt(0.0 * 0.0),
                      sqrt(0.0 * 0.0), sqrt(0.25 * 0.0)])
        self.assertEqual(fidelity.fidelity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                           q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_bhattacharyya(self):
        answer = - log(sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4), sqrt(0.625 * 0.5), sqrt(0.0 * 0.0),
                            sqrt(0.0 * 0.0), sqrt(0.25 * 0.0)]))
        self.assertEqual(fidelity.bhattacharyya(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_hellinger(self):
        answer = sqrt(
            2 * sum([(sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2, (sqrt(0.25) - sqrt(0.4)) ** 2,
                     (sqrt(0.625) - sqrt(0.5)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
                     (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.25) - sqrt(0.0)) ** 2]))
        self.assertEqual(fidelity.hellinger(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                            q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_matusita(self):
        answer = sqrt(
            sum([(sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2, (sqrt(0.25) - sqrt(0.4)) ** 2,
                 (sqrt(0.625) - sqrt(0.5)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
                 (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.25) - sqrt(0.0)) ** 2]))
        self.assertEqual(fidelity.matusita(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                           q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_squared_chord_similarity(self):
        answer = 2 * sum([sqrt(0.389 * 0.35), sqrt(0.5 * 0.563), sqrt(0.25 * 0.4), sqrt(0.625 * 0.5), sqrt(0.0 * 0.0),
                          sqrt(0.0 * 0.0), sqrt(0.25 * 0.0)]) - 1
        self.assertEqual(fidelity.squared_chord_similarity(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                           q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)

    def test_squared_chord_divergence(self):
        answer = sum([(sqrt(0.389) - sqrt(0.35)) ** 2, (sqrt(0.5) - sqrt(0.563)) ** 2, (sqrt(0.25) - sqrt(0.4)) ** 2,
                      (sqrt(0.625) - sqrt(0.5)) ** 2, (sqrt(0.0) - sqrt(0.0)) ** 2,
                      (sqrt(0.0) - sqrt(0.0)) ** 2, (sqrt(0.25) - sqrt(0.0)) ** 2])
        self.assertEqual(fidelity.squared_chord_divergence(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                                           q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer)
