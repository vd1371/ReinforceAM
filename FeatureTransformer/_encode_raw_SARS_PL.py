import numpy as np

from ._evaluate_rewards import _evaluate_rewards
from ._develop_array_of_states import _develop_array_of_states

def _encode_raw_SARS_PL(s_a_rs, valid_A = None, enough_budget = True, n_elements = 3):
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
		for id_ in s_a_rs:

			s_temp = []
			s_temp = _develop_array_of_states(s_temp, s_a_rs[id_])

			s[id_] = [s_temp,
						s_temp,
						s_temp]

			r[id_] = _evaluate_rewards(s_a_rs, id_, n_elements, valid_A, enough_budget)
			ac[id_] = np.sum(s_a_rs[id_]['elements_costs'])
			uc[id_] = s_a_rs[id_]['user_costs']

		return s, r, ac, uc