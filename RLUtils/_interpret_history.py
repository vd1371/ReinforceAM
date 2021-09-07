# Loading dependencies
import numpy as np

def interpret_history(vals, discount_vec, divide_by_n_elements = None):
	
	new_val = 0
	for id_ in vals.keys():
		for ne, it in enumerate(vals[id_]):
			new_val += np.sum(it * discount_vec)

	if not divide_by_n_elements is None:
		new_val /= divide_by_n_elements

	return new_val
