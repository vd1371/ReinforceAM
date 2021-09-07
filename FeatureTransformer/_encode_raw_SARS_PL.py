import numpy as np

from ._evaluate_rewards import _evaluate_rewards

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

			s_temp = list((s_a_rs[id_]['elements_age']-20)/20)
			s_temp += list((s_a_rs[id_]['elements_conds'] - 4.5) / 4.5)
			s_temp.append(s_a_rs[id_]['remaining_budget'])
			s_temp.append((s_a_rs[id_]['step']-10)/10)
			s_temp.append(s_a_rs[id_]['deviation'])

			s[id_] = [s_temp,
						s_temp,
						s_temp]

			r[id_] = _evaluate_rewards(s_a_rs, id_, n_elements, valid_A, enough_budget)
			ac[id_] = np.sum(s_a_rs[id_]['elements_costs'])
			uc[id_] = s_a_rs[id_]['user_costs']

		return s, r, ac, uc