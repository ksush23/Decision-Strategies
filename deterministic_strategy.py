import numpy as np
from random import uniform


def get_result(size):

    # get one (biggest) probability for deterministic strategy
    result = np.array([uniform(0.51, 1) for i in range(size)])
    return result
