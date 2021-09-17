import numpy as np

def _get_actions_of_fixed_plan(LrnObjs,
									fixed_plan,
									step):

	A = {}
	n_elements = LrnObjs.settings.n_elements
	for id_ in fixed_plan.keys():
		plan = [fixed_plan[id_][ne][step] for ne in range(n_elements)]
		A[id_] = np.array(plan)

	return A