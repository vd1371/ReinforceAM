import numpy as np

from ._develop_array_of_states import _develop_array_of_states

def _encode_raw_SARS_PL(s_a_rs, n_elements = 3):
		'''encoding the features, actions, and rewards
	
		valid_A: [True,True,True]
		enough_budget: Boolean
		'''

		s = {} # States
		ut = {} # rewards
		uc = {} # user costs
		ac = {} # agency costs
		for id_ in s_a_rs:

			s_temp = []
			s_temp = _develop_array_of_states(s_temp, s_a_rs[id_])

			s[id_] = [s_temp,
						s_temp,
						s_temp]

			ut[id_] = s_a_rs[id_]['elements_utils']
			ac[id_] = s_a_rs[id_]['elements_costs']
			uc[id_] = s_a_rs[id_]['user_costs']

		return s, ut, ac, uc