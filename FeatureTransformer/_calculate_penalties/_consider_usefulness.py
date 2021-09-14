

def _consider_usefulness(P_hist, ut_hist, uc_hist, A_hist, LrnObjs, penalty_val):

	# Taking care of those actions with no influence
	for id_ in A_hist.keys():
		for ne in range(LrnObjs.n_elements):
			for step in range(LrnObjs.settings.n_steps):

				if uc_hist[id_][ne][step] > 0 and \
					not ut_hist[id_][ne][step] > 0 and\
					not P_hist[id_][ne][step] < 0:

					P_hist[id_][ne][step] = penalty_val * \
												(A_hist[id_][ne][step] / 3)

	return P_hist