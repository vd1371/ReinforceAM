import numpy as np

def get_actions(S = None, LrnObjs = None, fixed_plan = None, step = None):
	# Finding the action

	n_elements = LrnObjs.settings.n_elements
	
	A = {}
	for id_ in LrnObjs.env.asset_IDs:
		if fixed_plan is None:
			A[id_] = LrnObjs.models[id_].predict_actions(S[id_],
														eps = LrnObjs.eps)
		else:
			plan = [fixed_plan[id_][ne][step] for ne in range(n_elements)]
			A[id_] = np.array(plan)

	return A