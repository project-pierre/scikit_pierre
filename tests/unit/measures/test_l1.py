import unittest
from math import log

from ....scikit_pierre.measures import l1


class TestL1(unittest.TestCase):
    def test_sorensen(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum([0.389 + 0.35, 0.5 + 0.563, 0.25 + 0.4, 0.625 + 0.5,
                           0.0 + 0.0, 0.0 + 0.0, 0.25 + 0.0])
        self.assertEqual(l1.sorensen(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                     q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_gower(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = 7
        self.assertEqual(l1.gower(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                  q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_soergel(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum(
            [max([0.389, 0.35]), max([0.5, 0.563]), max([0.25, 0.4]), max([0.625, 0.5]),
             max([0.0, 0.0]), max([0.0, 0.0]), max([0.25, 0.0])])
        self.assertEqual(l1.soergel(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                    q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_kulczynski_d(self):
        answer_num = sum([abs(0.389 - 0.35), abs(0.5 - 0.563), abs(0.25 - 0.4), abs(0.625 - 0.5),
                          abs(0.0 - 0.0), abs(0.0 - 0.0), abs(0.25 - 0.0)])
        answer_deno = sum(
            [min([0.389, 0.35]), min([0.5, 0.563]), min([0.25, 0.4]), min([0.625, 0.5]),
             min([0.0, 0.0]), min([0.0, 0.0]), min([0.25, 0.0])])
        self.assertEqual(l1.kulczynski_d(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                         q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num / answer_deno)

    def test_canberra(self):
        answer_num = sum(
            [abs(0.389 - 0.35) / (0.389 + 0.35), abs(0.5 - 0.563) / (0.5 + 0.563),
             abs(0.25 - 0.4) / (0.25 + 0.4),
             abs(0.625 - 0.5) / (0.625 + 0.5), abs(0.0 - 0.0) / 0.00001, abs(0.0 - 0.0) / 0.00001,
             abs(0.25 - 0.0) / (0.25 + 0.0)])
        self.assertEqual(l1.canberra(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                     q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)

    def test_lorentzian(self):
        answer_num = sum(
            [log(1 + abs(0.389 - 0.35)), log(1 + abs(0.5 - 0.563)), log(1 + abs(0.25 - 0.4)),
             log(1 + abs(0.625 - 0.5)), log(1 - 0), log(1 - 0),
             log(1 + abs(0.25 - 0.0))])
        self.assertEqual(l1.lorentzian(p=[0.389, 0.5, 0.25, 0.625, 0.0, 0.0, 0.25],
                                       q=[0.35, 0.563, 0.4, 0.5, 0.0, 0.0, 0.0]),
                         answer_num)
