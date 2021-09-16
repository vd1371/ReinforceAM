import numpy as np
import itertools
from .._MRRHolders import PossibleMRRforIUC

def get_all_possible_options(env, step):

	all_options = []
	for id_ in env.asset_IDs:
		for possible_mrr in itertools.product([0, 1, 2, 3], repeat = 3):

			ac_costs, uc_costs, utils, step = env.simulators[id_].cost_of(possible_mrr)
			all_options.append(PossibleMRRforIUC(ID = id_,
													mrr = possible_mrr,
													ac_costs = ac_costs,
													uc_costs = uc_costs,
													utils = utils,
													step = step))
	return all_options