import numpy as np
import pandas as pd

def _find_summary_of_network_states(s_a_rs, n_elements):

	s_common = []

	ages = []
	conds = []
	devs = []

	for id_ in s_a_rs:
		ages.append((s_a_rs[id_]['elements_age']-20)/40)
		conds.append((s_a_rs[id_]['elements_conds'] - 4.5) / 4.5)
		devs.append(s_a_rs[id_]['deviation'])

	ages = pd.DataFrame(np.array(ages))
	conds = pd.DataFrame(np.array(conds))
	devs = pd.DataFrame(np.array(devs))

	s_common += ages.describe().iloc[1:, :].values.flatten().tolist()
	s_common += conds.describe().iloc[1:, :].values.flatten().tolist()
	s_common += devs.describe().iloc[1:, :].values.flatten().tolist()

	s_common.append(s_a_rs[id_]['remaining_budget'])
	s_common.append((s_a_rs[id_]['step']-10)/10)

	return s_common

