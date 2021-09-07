import numpy as np
from copy import deepcopy

def update_actions_with_new_acitons(previous_actions,
									best_actions_at,
									step,
									n_elements):

	updated_actions = deepcopy(previous_actions)
	for id_ in best_actions_at:

		for ne in range(n_elements):
			updated_actions[id_][ne, step] = best_actions_at[id_][ne]

	return updated_actions