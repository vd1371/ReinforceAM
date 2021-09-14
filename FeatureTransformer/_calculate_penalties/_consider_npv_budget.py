

def _consider_npv_budget(P_hist, S_hist, LrnObjs, penalty_val):

	first_asset_id_ = list(S_hist.keys())[0]
	remaining_budget_ind = -3
	first_element = 0
	for step in range(LrnObjs.settings.n_steps):
		enough_budget = S_hist[first_asset_id_][first_element][step][remaining_budget_ind] > 0
		if not enough_budget:
			for id_ in S_hist.keys():
				P_hist[id_][:, step:] = penalty_val
			break

	return P_hist