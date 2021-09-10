# Loading dependencies
import numpy as np

def discount(vec, discount_vec):

	return np.sum(vec * discount_vec)
