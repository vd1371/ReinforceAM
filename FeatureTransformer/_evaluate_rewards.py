import numpy as np

def _evaluate_rewards(s_a_rs, id_, n_elements, valid_A, enough_budget):

	R = s_a_rs[id_]['elements_utils']/ \
					(s_a_rs[id_]['elements_costs']+ \
						s_a_rs[id_]['user_costs'])**0.2
			
	for elem in range(n_elements):
		if s_a_rs[id_]['elements_costs'][elem] > 0:

			if s_a_rs[id_]['elements_utils'][elem] == 0:
				R[elem] = 0

			if not enough_budget or \
					not valid_A[id_][elem]:
				R[elem] = -5

	R[np.isnan(R)] = 0

	return R