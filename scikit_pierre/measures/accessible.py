from . import minkowski, l1, intersection, inner_product, shannon, fidelity, chi, combinations, vicissitude


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
    elif measure == "CHEBYSHEV":
        return minkowski.chebyshev
    # L1 Family
    elif measure == "SORESEN":
        return l1.sorensen
    elif measure == "GOWER":
        return l1.gower
    elif measure == "SOERGEL":
        return l1.soergel
    elif measure == "KULCZYNSKI_D":
        return l1.kulczynski_d
    elif measure == "CANBERRA":
        return l1.canberra
    elif measure == "LORENTZIAN":
        return l1.lorentzian
    # Intersection Family
    elif measure == "INTERSECTION_SIM":
        return intersection.intersection_similarity
    elif measure == "INTERSECTION_DIV":
        return intersection.intersection_divergence
    elif measure == "WAVE":
        return intersection.wave_hedges
    elif measure == "CZEKANOWSKI_SIM":
        return intersection.czekanowski_similarity
    elif measure == "CZEKANOWSKI_DIV":
        return intersection.czekanowski_divergence
    elif measure == "MOTYKA_SIM":
        return intersection.motyka_similarity
    elif measure == "MOTYKA_DIV":
        return intersection.motyka_divergence
    elif measure == "KULCZYNSKI_S":
        return intersection.kulczynski_s
    elif measure == "RUZICKA":
        return intersection.ruzicka
    elif measure == "TONIMOTO":
        return intersection.tanimoto
    # Inner Production Family
    elif measure == "INNER":
        return inner_product.inner_product
    elif measure == "HARMONIC":
        return inner_product.harmonic_mean
    elif measure == "COSINE":
        return inner_product.cosine
    elif measure == "KUMAR_HASSEBROOK":
        return inner_product.kumar_hassebrook
    elif measure == "JACCARD":
        return inner_product.jaccard
    elif measure == "DICE_SIM":
        return inner_product.dice_similarity
    elif measure == "DICE_DIV":
        return inner_product.dice_divergence
    # Fidelity Family
    elif measure == "FIDELITY":
        return fidelity.fidelity
    elif measure == "BHATTACHARYYA":
        return fidelity.bhattacharyya
    elif measure == "HELLINGER":
        return fidelity.hellinger
    elif measure == "MATUSITA":
        return fidelity.matusita
    elif measure == "SQUARED_CHORD_SIM":
        return fidelity.squared_chord_similarity
    elif measure == "SQUARED_CHORD_DIV":
        return fidelity.squared_chord_divergence
    # Chi Square Family
    elif measure == "SQUARED_EUCLIDEAN":
        return chi.squared_euclidean
    elif measure == "CHI_SQUARE":
        return chi.person_chi_square
    elif measure == "NEYMAN":
        return chi.neyman_square
    elif measure == "SQUARED_CHI":
        return chi.squared_chi_square
    elif measure == "PROBABILISTIC_CHI":
        return chi.probabilistic_symmetric_chi_square
    elif measure == "DIVERGENCE":
        return chi.divergence
    elif measure == "CLARK":
        return chi.clark
    elif measure == "ADDITIVE_CHI":
        return chi.additive_symmetric_chi_squared
    # Shannon's Entropy Family
    elif measure == "KL":
        return shannon.kullback_leibler
    elif measure == "JEFFREYS":
        return shannon.jeffreys
    elif measure == "K_DIV":
        return shannon.k_divergence
    elif measure == "TOPSOE":
        return shannon.topsoe
    elif measure == "JENSEN_SHANNON":
        return shannon.jensen_shannon
    elif measure == "JENSEN_DIFF":
        return shannon.jensen_difference
    # Combinations
    elif measure == "TANEJA":
        return combinations.taneja
    elif measure == "KUMAR_JOHNSON":
        return combinations.kumar_johnson
    elif measure == "AVG":
        return combinations.avg
    elif measure == "WTV":
        return combinations.weighted_total_variation
    # Vicissitude
    elif measure == "VICIS_WAVE":
        return vicissitude.vicis_wave_hedges
    elif measure == "VICIS_EMANON2":
        return vicissitude.vicis_symmetric_chi_square
    elif measure == "VICIS_EMANON3":
        return vicissitude.vicis_symmetric_chi_square_emanon3
    elif measure == "VICIS_EMANON4":
        return vicissitude.vicis_symmetric_chi_square_emanon4
    elif measure == "VICIS_EMANON5":
        return vicissitude.max_symmetric_chi_square_emanon5
    elif measure == "VICIS_EMANON6":
        return vicissitude.min_symmetric_chi_square_emanon6


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
