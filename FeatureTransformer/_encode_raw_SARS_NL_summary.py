import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from ._evaluate_rewards import _evaluate_rewards
from ._find_summary_of_network_states import _find_summary_of_network_states

def _encode_raw_SARS_NL_summary(s_a_rs, 
									valid_A = None,
									enough_budget = True,
									n_elements = 3):
		'''encoding the features, actions, and rewards
	
		valid_A: [True,True,True]
		enough_budget: Boolean
		'''
		if valid_A is None:
			valid_A = [True, True, True]

		s = {} # States
		r = {} # rewards
		uc = {} # user costs
		ac = {} # agency costs

		s_common = _find_summary_of_network_states(s_a_rs, n_elements)

		s_temp = []
		for id_ in s_a_rs:

			s_temp = s_common[:]
			s_temp += list((s_a_rs[id_]['elements_age']-20)/20)
			s_temp += list((s_a_rs[id_]['elements_conds'] - 4.5) / 4.5)

			# Finding rewards
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
			r[id_] = R

			# Finding Agency costs and user costs
			ac[id_] = np.sum(s_a_rs[id_]['elements_costs'])
			uc[id_] = s_a_rs[id_]['user_costs']

			s[id_] = [s_temp, s_temp, s_temp]

		return s, r, ac, uc