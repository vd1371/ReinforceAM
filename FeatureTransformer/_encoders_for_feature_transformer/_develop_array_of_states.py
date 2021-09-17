from ._encode_age import _encode_age
from ._encode_conditions import _encode_conditions
from ._encode_steps import _encode_steps
from ._encode_deviation import _encode_deviation


def _develop_array_of_states(inp, s_a_rs_id):

	inp += _encode_age(s_a_rs_id['elements_age'])
	inp += _encode_conditions(s_a_rs_id['elements_conds'])
	inp.append(s_a_rs_id['remaining_budget'])
	inp.append(_encode_steps(s_a_rs_id['step']))
	inp.append(_encode_deviation(s_a_rs_id['deviation']))

	return inp