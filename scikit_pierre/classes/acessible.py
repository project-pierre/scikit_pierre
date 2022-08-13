from .genre import genre_probability_approach


def class_funcs(class_approach: str = "GENRE_PROBABILITY"):
    if class_approach == "GENRE_PROBABILITY":
        return genre_probability_approach
    else:
        raise Exception("Class approach not found!")
