import numpy as np

def get_all_actions_of_step(all_actions, step, n_elements):

	actions = {}
	for id_ in all_actions:
		actions[id_] = all_actions[id_][np.arange(n_elements), step]

	return actions