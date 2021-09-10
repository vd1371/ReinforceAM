import numpy as np
from copy import deepcopy

def calculate_penalties(LrnObjs):

	S_hist, A_hist, ut_hist, \
			nextS_hist, ac_hist, uc_hist = LrnObjs.episode_holder.get()

	remaining_budget_ind = -3

	penalty_val = -5

	# The penalties are based on the validation of actions
	P_hist = deepcopy(LrnObjs.validator.A_valid_hist)
	for id_ in P_hist:
		P_hist[id_][P_hist[id_] == 0] = penalty_val
		P_hist[id_][P_hist[id_] == 1] = 0

	## Taking care of budget
	first_asset_id_ = list(S_hist.keys())[0]
	first_element = 0
	for step in range(LrnObjs.settings.n_steps):
		enough_budget = S_hist[first_asset_id_][first_element][step][remaining_budget_ind] > 0
		if not enough_budget:
			for id_ in S_hist.keys():
				P_hist[id_][:, step:] = penalty_val
			break

	# Taking care of those actions with no influence
	for id_ in S_hist.keys():
		for ne in range(LrnObjs.n_elements):
			for step in range(LrnObjs.settings.n_steps):

				if uc_hist[id_][ne][step] > 0 and \
					not ut_hist[id_][ne][step] > 0:

					P_hist[id_][ne][step] = penalty_val * \
												(A_hist[id_][ne][step] / 3)

	return P_hist
			


