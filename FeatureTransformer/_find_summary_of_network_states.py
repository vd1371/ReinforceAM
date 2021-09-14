import numpy as np
import pandas as pd

from ._encode_age import _encode_age
from ._encode_conditions import _encode_conditions
from ._encode_deviation import _encode_deviation


def _find_summary_of_network_states(s_a_rs, n_elements):

	ages = []
	conds = []
	devs = []

	for id_ in s_a_rs:
		ages.append(_encode_age(s_a_rs[id_]['elements_age']))
		conds.append(_encode_conditions(s_a_rs[id_]['elements_conds']))
		devs.append(_encode_deviation(s_a_rs[id_]['deviation']))

	ages = pd.DataFrame(np.array(ages))
	conds = pd.DataFrame(np.array(conds))
	devs = pd.DataFrame(np.array(devs))

	s_common = []
	
	s_common += ages.describe().iloc[1:, :].values.flatten().tolist()
	s_common += conds.describe().iloc[1:, :].values.flatten().tolist()
	s_common += devs.describe().iloc[1:, :].values.flatten().tolist()

	return s_common

