import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from ._encode_age import _encode_age
from ._encode_conditions import _encode_conditions
from ._encode_steps import _encode_steps
from ._encode_deviation import _encode_deviation

def _encode_raw_SARS_NL_detailed(s_a_rs):
		'''encoding the features, actions, and rewards

		enough_budget: Boolean
		'''

		s = {} # States
		ut = {} # rewards
		uc = {} # user costs
		ac = {} # agency costs

		s_temp = []
		first = True
		for id_ in s_a_rs:

			if first:
				# remaining_budget and steo are shared among all
				# No point for putting them in the state vector
				s_temp.append(s_a_rs[id_]['remaining_budget'])
				s_temp.append(_encode_steps(s_a_rs[id_]['step']))
				first = False

			s_temp += _encode_age(s_a_rs[id_]['elements_age'])
			s_temp += _encode_conditions(s_a_rs[id_]['elements_conds'])
			s_temp.append(_encode_deviation(s_a_rs[id_]['deviation']))

			ut[id_] = s_a_rs[id_]['elements_utils']
			ac[id_] = s_a_rs[id_]['elements_costs']
			uc[id_] = s_a_rs[id_]['user_costs']

		for id_ in s_a_rs:
			s[id_] = [s_temp, s_temp, s_temp]

		return s, ut, ac, uc