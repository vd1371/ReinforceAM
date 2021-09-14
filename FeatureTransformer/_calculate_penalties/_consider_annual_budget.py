
def _consider_annual_budget(P_hist, enough_annual_budget, LrnObjs, penalty_val):

	for step, enough_budget in zip(range(LrnObjs.settings.n_steps), enough_annual_budget):
		if not enough_budget:
			for id_ in P_hist.keys():
				P_hist[id_][:, step] = penalty_val

	return P_hist