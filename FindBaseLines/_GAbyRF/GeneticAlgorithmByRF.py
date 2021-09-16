import numpy as np

from ._find_all_optimized_actions import find_all_optimized_actions
from ._find_cost_of_plans_at_step import find_cost_of_plans_at_step
from ._defer_optimized_actions_with import defer_optimized_actions_with

from .._IUC import sort_and_optimize
from .._IUC import update_actions_with_new_acitons
from .._IUC import get_all_actions_of_step

from .._MRRHolders import MCMRRHolder


def GeneticAlgorithmByRF(LearningObjects, base_direc, n_assets):

	LrnObjs = LearningObjects(base_direc = base_direc,
								n_assets = n_assets)

	budget_model = LrnObjs.env.annual_budget_limit
	IDs = LrnObjs.env.asset_IDs
	n_steps = LrnObjs.settings.n_steps
	n_elements = LrnObjs.n_elements

	all_optimzied_actions = find_all_optimized_actions(**LrnObjs.__dict__)

	actions = {id_: np.zeros((n_elements, n_steps)) for id_ in IDs}
	for step in range(n_steps):

		print (f"Step {step} of the GeneticAlgorithmByRF is about to analyzed")

		optimized_actions_costs = find_cost_of_plans_at_step(all_optimzied_actions,
															step,
															**LrnObjs.__dict__)

		best_actions_at = sort_and_optimize(optimized_actions_costs,
												budget = budget_model,
												step = step)

		actions = update_actions_with_new_acitons(previous_actions = actions,
												best_actions_at = best_actions_at,
												step = step,
												n_elements = n_elements)

		actions_of_step = get_all_actions_of_step(actions,
													step = step,
													n_elements = n_elements)

		LrnObjs.env.step(actions_of_step)

		all_optimzied_actions = defer_optimized_actions_with(all_optimzied_actions,
															best_actions_at,
															step)
	# save the results
	mrr_holders = MCMRRHolder(**LrnObjs.__dict__)
	mrr_holders.update(actions)
	mrr_holders.get_baseline()

	LrnObjs.env.reset()
	
	return actions