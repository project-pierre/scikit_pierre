"""
This file contains the call to all similarity and divergence measure.
"""

from . import minkowski, l1, intersection, inner_product, shannon, fidelity, chi, combinations, \
    vicissitude


def calibration_measures_funcs(measure: str = "KL"):
    """
    Function to decide what distance measure will be used.

    :param measure: The acronyms (initials) assigned to a distance measure, which will be used by.
    :return: The choose function.
    """
    # Minkowski Family
    if measure == "MINKOWSKI":
        return minkowski.minkowski
    if measure == "EUCLIDEAN":
        return minkowski.euclidean
    if measure == "CITY_BLOCK":
        return minkowski.city_block
    if measure == "CHEBYSHEV":
        return minkowski.chebyshev
    # L1 Family
    if measure == "SORESEN":
        return l1.sorensen
    if measure == "GOWER":
        return l1.gower
    if measure == "SOERGEL":
        return l1.soergel
    if measure == "KULCZYNSKI_D":
        return l1.kulczynski_d
    if measure == "CANBERRA":
        return l1.canberra
    if measure == "LORENTZIAN":
        return l1.lorentzian
    # Intersection Family
    if measure == "INTERSECTION_SIM":
        return intersection.intersection_similarity
    if measure == "INTERSECTION_DIV":
        return intersection.intersection_divergence
    if measure == "WAVE":
        return intersection.wave_hedges
    if measure == "CZEKANOWSKI_SIM":
        return intersection.czekanowski_similarity
    if measure == "CZEKANOWSKI_DIV":
        return intersection.czekanowski_divergence
    if measure == "MOTYKA_SIM":
        return intersection.motyka_similarity
    if measure == "MOTYKA_DIV":
        return intersection.motyka_divergence
    if measure == "KULCZYNSKI_S":
        return intersection.kulczynski_s
    if measure == "RUZICKA":
        return intersection.ruzicka
    if measure == "TONIMOTO":
        return intersection.tanimoto
    # Inner Production Family
    if measure == "INNER":
        return inner_product.inner_product
    if measure == "HARMONIC":
        return inner_product.harmonic_mean
    if measure == "COSINE":
        return inner_product.cosine
    if measure == "KUMAR_HASSEBROOK":
        return inner_product.kumar_hassebrook
    if measure == "JACCARD":
        return inner_product.jaccard
    if measure == "DICE_SIM":
        return inner_product.dice_similarity
    if measure == "DICE_DIV":
        return inner_product.dice_divergence
    # Fidelity Family
    if measure == "FIDELITY":
        return fidelity.fidelity
    if measure == "BHATTACHARYYA":
        return fidelity.bhattacharyya
    if measure == "HELLINGER":
        return fidelity.hellinger
    if measure == "MATUSITA":
        return fidelity.matusita
    if measure == "SQUARED_CHORD_SIM":
        return fidelity.squared_chord_similarity
    if measure == "SQUARED_CHORD_DIV":
        return fidelity.squared_chord_divergence
    # Chi Square Family
    if measure == "SQUARED_EUCLIDEAN":
        return chi.squared_euclidean
    if measure == "CHI_SQUARE":
        return chi.person_chi_square
    if measure == "NEYMAN":
        return chi.neyman_square
    if measure == "SQUARED_CHI":
        return chi.squared_chi_square
    if measure == "PROBABILISTIC_CHI":
        return chi.probabilistic_symmetric_chi_square
    if measure == "DIVERGENCE":
        return chi.divergence
    if measure == "CLARK":
        return chi.clark
    if measure == "ADDITIVE_CHI":
        return chi.additive_symmetric_chi_squared
    # Shannon's Entropy Family
    if measure == "KL":
        return shannon.kullback_leibler
    if measure == "JEFFREYS":
        return shannon.jeffreys
    if measure == "K_DIV":
        return shannon.k_divergence
    if measure == "TOPSOE":
        return shannon.topsoe
    if measure == "JENSEN_SHANNON":
        return shannon.jensen_shannon
    if measure == "JENSEN_DIFF":
        return shannon.jensen_difference
    # Combinations
    if measure == "TANEJA":
        return combinations.taneja
    if measure == "KUMAR_JOHNSON":
        return combinations.kumar_johnson
    if measure == "AVG":
        return combinations.avg
    if measure == "WTV":
        return combinations.weighted_total_variation
    # Vicissitude
    if measure == "VICIS_WAVE":
        return vicissitude.vicis_wave_hedges
    if measure == "VICIS_EMANON2":
        return vicissitude.vicis_symmetric_chi_square
    if measure == "VICIS_EMANON3":
        return vicissitude.vicis_symmetric_chi_square_emanon3
    if measure == "VICIS_EMANON4":
        return vicissitude.vicis_symmetric_chi_square_emanon4
    if measure == "VICIS_EMANON5":
        return vicissitude.max_symmetric_chi_square_emanon5
    if measure == "VICIS_EMANON6":
        return vicissitude.min_symmetric_chi_square_emanon6
    raise NameError(f"Measure not found! {measure}")


SIMILARITY_LIST = [
    # Intersection
    'INTERSECTION_SIM',
    'CZEKANOWSKI_SIM',
    'MOTYKA_SIM',
    'KULCZYNSKI_S',
    'RUZICKA',
    # Inner Production Family
    'INNER',
    'HARMONIC',
    'COSINE',
    'KUMAR',
    'DICE_SIM',
    # Fidelity Family
    'FIDELITY',
    'SQUARED_CHORD_SIM',
]
