import numpy as np

def defer_optimized_actions_with(all_optimzied_actions,
								best_actions_at,
								step):

	for id_ in all_optimzied_actions:
		if not id_ in best_actions_at and \
			not np.all(all_optimzied_actions[id_][[0, 1 , 2], step] == 0):

				all_optimzied_actions[id_] = np.roll(all_optimzied_actions[id_], 1)
	
	return all_optimzied_actions