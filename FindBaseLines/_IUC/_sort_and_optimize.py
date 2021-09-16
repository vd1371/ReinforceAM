import numpy as np
from operator import itemgetter
import time

def sort_and_optimize(all_options, budget, step):

	remaining_budget = budget[step]

	start = time.time()
	all_options.sort(key=lambda x: x.u_c, reverse=True)

	selected_assets = []
	best_actions_at = {}

	for possible_mrr in all_options:

		if remaining_budget > possible_mrr.ac_costs:
			if not possible_mrr.ID in selected_assets and \
					possible_mrr.utils > 0:
				remaining_budget -= np.sum(possible_mrr.ac_costs)
				selected_assets.append(possible_mrr.ID)
				best_actions_at[possible_mrr.ID] = possible_mrr.mrr

		else:
			break

	return best_actions_at


