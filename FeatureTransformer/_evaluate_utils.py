import numpy as np

def _evaluate_utils(s_a_rs, id_, n_elements, valid_A, enough_budget):
	
	ut = s_a_rs[id_]['elements_utils']

	for elem in range(n_elements):

		# If any action has been taken
		if s_a_rs[id_]['elements_costs'][elem] > 0:

			if s_a_rs[id_]['elements_utils'][elem] == 0:
				R[elem] = 0

			if not enough_budget or \
					not valid_A[id_][elem]:
				R[elem] = -30

	R[np.isnan(R)] = 0

	return R