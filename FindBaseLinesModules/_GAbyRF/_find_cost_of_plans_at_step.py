import numpy as np
from .._MRRHolders import PossibleMRRforIUC

def find_cost_of_plans_at_step(optimized_plans, step, **params):

	env = params.pop("env")
	n_elements = params.pop("n_elements")

	holder = []
	for id_ in optimized_plans:

		asset_mrr = optimized_plans[id_][np.arange(n_elements), step]

		ac_costs, uc_costs, utils, step = env.simulators[id_].cost_of(asset_mrr)
		holder.append(PossibleMRRforIUC(ID = id_,
												mrr = asset_mrr,
												ac_costs = ac_costs,
												uc_costs = uc_costs,
												utils = utils,
												step = step))

	return holder

