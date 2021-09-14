import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from ._find_summary_of_network_states import _find_summary_of_network_states
from ._develop_array_of_states import _develop_array_of_states

from ._encode_age import _encode_age
from ._encode_conditions import _encode_conditions
from ._encode_steps import _encode_steps
from ._encode_deviation import _encode_deviation

def _encode_raw_SARS_NL_summary(s_a_rs, n_elements = 3):
		'''encoding the features, actions, and rewards
	
		valid_A: [True,True,True]
		enough_budget: Boolean
		'''

		s = {} # States
		ut = {} # rewards
		uc = {} # user costs
		ac = {} # agency costs

		s_common = _find_summary_of_network_states(s_a_rs, n_elements)

		for id_ in s_a_rs:

			s_temp = s_common[:]

			s_temp += _encode_age(s_a_rs[id_]['elements_age'])
			s_temp += _encode_conditions(s_a_rs[id_]['elements_conds'])
			s_temp.append(_encode_steps(s_a_rs[id_]['step']))
			
			s[id_] = [s_temp, s_temp, s_temp]

			ut[id_] = s_a_rs[id_]['elements_utils']
			ac[id_] = s_a_rs[id_]['elements_costs']
			uc[id_] = s_a_rs[id_]['user_costs']

		return s, ut, ac, uc