import numpy as np
import time

from ._update_actions_with_new_acitons import update_actions_with_new_acitons
from ._sort_and_optimize import sort_and_optimize
from ._get_all_possible_options import get_all_possible_options
from ._get_all_actions_of_step import get_all_actions_of_step

def get_one_IUC_instance(**params):

	env = params.pop('env')
	IDs = env.asset_IDs
	budget_model = env.annual_budget_limit
	n_elements = params.pop('n_elements')
	n_steps = params.pop("settings").n_steps
	dim_actions = params.pop("dim_actions")
	logger = params.pop("logger")


	actions = {id_: np.zeros((n_elements, n_steps)) for id_ in IDs}
	for step in range(n_steps):

		start = time.time()
		all_options = get_all_possible_options(env, step)

		best_actions_at = sort_and_optimize(all_options,
												budget = budget_model,
												step = step)

		actions = update_actions_with_new_acitons(previous_actions = actions,
												best_actions_at = best_actions_at,
												step = step,
												n_elements = n_elements)

		actions_of_step = get_all_actions_of_step(actions,
													step = step,
													n_elements = n_elements)

		env.step(actions_of_step)

	env.reset()
	return actions