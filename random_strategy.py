from random import random
import numpy as np


def get_result(size):

    # get random probabilities for each class
    result = np.array([np.array([random() for i in range(3)]) for j in range(size)])
    return result
