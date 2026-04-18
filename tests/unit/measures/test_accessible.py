"""
Unit Test Cases for the accessible module (calibration_measures_funcs dispatcher).
"""
import unittest

from scikit_pierre.measures import accessible
from scikit_pierre.measures import (chi, combinations, fidelity, inner_product,
                                    intersection, l1, minkowski, shannon, vicissitude)


class TestAccessible(unittest.TestCase):

    def test_returns_minkowski(self):
        self.assertIs(accessible.calibration_measures_funcs("MINKOWSKI"), minkowski.minkowski)

    def test_returns_euclidean(self):
        self.assertIs(accessible.calibration_measures_funcs("EUCLIDEAN"), minkowski.euclidean)

    def test_returns_city_block(self):
        self.assertIs(accessible.calibration_measures_funcs("CITY_BLOCK"), minkowski.city_block)

    def test_returns_chebyshev(self):
        self.assertIs(accessible.calibration_measures_funcs("CHEBYSHEV"), minkowski.chebyshev)

    def test_returns_sorensen(self):
        self.assertIs(accessible.calibration_measures_funcs("SORESEN"), l1.sorensen)

    def test_returns_gower(self):
        self.assertIs(accessible.calibration_measures_funcs("GOWER"), l1.gower)

    def test_returns_soergel(self):
        self.assertIs(accessible.calibration_measures_funcs("SOERGEL"), l1.soergel)

    def test_returns_kulczynski_d(self):
        self.assertIs(accessible.calibration_measures_funcs("KULCZYNSKI_D"), l1.kulczynski_d)

    def test_returns_canberra(self):
        self.assertIs(accessible.calibration_measures_funcs("CANBERRA"), l1.canberra)

    def test_returns_lorentzian(self):
        self.assertIs(accessible.calibration_measures_funcs("LORENTZIAN"), l1.lorentzian)

    def test_returns_intersection_similarity(self):
        self.assertIs(accessible.calibration_measures_funcs("INTERSECTION_SIM"),
                      intersection.intersection_similarity)

    def test_returns_intersection_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("INTERSECTION_DIV"),
                      intersection.intersection_divergence)

    def test_returns_wave_hedges(self):
        self.assertIs(accessible.calibration_measures_funcs("WAVE"), intersection.wave_hedges)

    def test_returns_czekanowski_similarity(self):
        self.assertIs(accessible.calibration_measures_funcs("CZEKANOWSKI_SIM"),
                      intersection.czekanowski_similarity)

    def test_returns_czekanowski_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("CZEKANOWSKI_DIV"),
                      intersection.czekanowski_divergence)

    def test_returns_motyka_similarity(self):
        self.assertIs(accessible.calibration_measures_funcs("MOTYKA_SIM"),
                      intersection.motyka_similarity)

    def test_returns_motyka_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("MOTYKA_DIV"),
                      intersection.motyka_divergence)

    def test_returns_kulczynski_s(self):
        self.assertIs(accessible.calibration_measures_funcs("KULCZYNSKI_S"),
                      intersection.kulczynski_s)

    def test_returns_ruzicka(self):
        self.assertIs(accessible.calibration_measures_funcs("RUZICKA"), intersection.ruzicka)

    def test_returns_tanimoto(self):
        self.assertIs(accessible.calibration_measures_funcs("TONIMOTO"), intersection.tanimoto)

    def test_returns_inner_product(self):
        self.assertIs(accessible.calibration_measures_funcs("INNER"), inner_product.inner_product)

    def test_returns_harmonic_mean(self):
        self.assertIs(accessible.calibration_measures_funcs("HARMONIC"), inner_product.harmonic_mean)

    def test_returns_cosine(self):
        self.assertIs(accessible.calibration_measures_funcs("COSINE"), inner_product.cosine)

    def test_returns_kumar_hassebrook(self):
        self.assertIs(accessible.calibration_measures_funcs("KUMAR_HASSEBROOK"),
                      inner_product.kumar_hassebrook)

    def test_returns_jaccard(self):
        self.assertIs(accessible.calibration_measures_funcs("JACCARD"), inner_product.jaccard)

    def test_returns_dice_similarity(self):
        self.assertIs(accessible.calibration_measures_funcs("DICE_SIM"), inner_product.dice_similarity)

    def test_returns_dice_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("DICE_DIV"), inner_product.dice_divergence)

    def test_returns_fidelity(self):
        self.assertIs(accessible.calibration_measures_funcs("FIDELITY"), fidelity.fidelity)

    def test_returns_bhattacharyya(self):
        self.assertIs(accessible.calibration_measures_funcs("BHATTACHARYYA"), fidelity.bhattacharyya)

    def test_returns_hellinger(self):
        self.assertIs(accessible.calibration_measures_funcs("HELLINGER"), fidelity.hellinger)

    def test_returns_matusita(self):
        self.assertIs(accessible.calibration_measures_funcs("MATUSITA"), fidelity.matusita)

    def test_returns_squared_chord_similarity(self):
        self.assertIs(accessible.calibration_measures_funcs("SQUARED_CHORD_SIM"),
                      fidelity.squared_chord_similarity)

    def test_returns_squared_chord_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("SQUARED_CHORD_DIV"),
                      fidelity.squared_chord_divergence)

    def test_returns_squared_euclidean(self):
        self.assertIs(accessible.calibration_measures_funcs("SQUARED_EUCLIDEAN"),
                      chi.squared_euclidean)

    def test_returns_chi_square(self):
        self.assertIs(accessible.calibration_measures_funcs("CHI_SQUARE"), chi.person_chi_square)

    def test_returns_neyman(self):
        self.assertIs(accessible.calibration_measures_funcs("NEYMAN"), chi.neyman_square)

    def test_returns_squared_chi(self):
        self.assertIs(accessible.calibration_measures_funcs("SQUARED_CHI"), chi.squared_chi_square)

    def test_returns_probabilistic_chi(self):
        self.assertIs(accessible.calibration_measures_funcs("PROBABILISTIC_CHI"),
                      chi.probabilistic_symmetric_chi_square)

    def test_returns_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("DIVERGENCE"), chi.divergence)

    def test_returns_clark(self):
        self.assertIs(accessible.calibration_measures_funcs("CLARK"), chi.clark)

    def test_returns_additive_chi(self):
        self.assertIs(accessible.calibration_measures_funcs("ADDITIVE_CHI"),
                      chi.additive_symmetric_chi_squared)

    def test_returns_kullback_leibler_default(self):
        self.assertIs(accessible.calibration_measures_funcs(), shannon.kullback_leibler)

    def test_returns_kullback_leibler_kl(self):
        self.assertIs(accessible.calibration_measures_funcs("KL"), shannon.kullback_leibler)

    def test_returns_jeffreys(self):
        self.assertIs(accessible.calibration_measures_funcs("JEFFREYS"), shannon.jeffreys)

    def test_returns_k_divergence(self):
        self.assertIs(accessible.calibration_measures_funcs("K_DIV"), shannon.k_divergence)

    def test_returns_topsoe(self):
        self.assertIs(accessible.calibration_measures_funcs("TOPSOE"), shannon.topsoe)

    def test_returns_jensen_shannon(self):
        self.assertIs(accessible.calibration_measures_funcs("JENSEN_SHANNON"),
                      shannon.jensen_shannon)

    def test_returns_jensen_difference(self):
        self.assertIs(accessible.calibration_measures_funcs("JENSEN_DIFF"),
                      shannon.jensen_difference)

    def test_returns_taneja(self):
        self.assertIs(accessible.calibration_measures_funcs("TANEJA"), combinations.taneja)

    def test_returns_kumar_johnson(self):
        self.assertIs(accessible.calibration_measures_funcs("KUMAR_JOHNSON"),
                      combinations.kumar_johnson)

    def test_returns_avg(self):
        self.assertIs(accessible.calibration_measures_funcs("AVG"), combinations.avg)

    def test_returns_wtv(self):
        self.assertIs(accessible.calibration_measures_funcs("WTV"),
                      combinations.weighted_total_variation)

    def test_returns_vicis_wave(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_WAVE"),
                      vicissitude.vicis_wave_hedges)

    def test_returns_vicis_emanon2(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_EMANON2"),
                      vicissitude.vicis_symmetric_chi_square)

    def test_returns_vicis_emanon3(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_EMANON3"),
                      vicissitude.vicis_symmetric_chi_square_emanon3)

    def test_returns_vicis_emanon4(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_EMANON4"),
                      vicissitude.vicis_symmetric_chi_square_emanon4)

    def test_returns_vicis_emanon5(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_EMANON5"),
                      vicissitude.max_symmetric_chi_square_emanon5)

    def test_returns_vicis_emanon6(self):
        self.assertIs(accessible.calibration_measures_funcs("VICIS_EMANON6"),
                      vicissitude.min_symmetric_chi_square_emanon6)

    def test_invalid_measure_raises_name_error(self):
        with self.assertRaises(NameError):
            accessible.calibration_measures_funcs("INVALID_MEASURE")

    def test_invalid_measure_error_message_contains_name(self):
        try:
            accessible.calibration_measures_funcs("UNKNOWN_XYZ")
        except NameError as e:
            self.assertIn("UNKNOWN_XYZ", str(e))

    def test_returned_function_is_callable(self):
        func = accessible.calibration_measures_funcs("KL")
        self.assertTrue(callable(func))

    def test_returned_function_produces_float(self):
        func = accessible.calibration_measures_funcs("KL")
        result = func([0.5, 0.5], [0.5, 0.5])
        self.assertIsInstance(result, float)
