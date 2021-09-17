import numpy as np

def get_average_of_rewards(R_hist, P_hist, LrnObjs):

	R_avg = 0
	for id_ in R_hist:
		R_avg = R_avg + np.sum(R_hist[id_]) + np.sum(P_hist[id_])
	R_avg = R_avg / LrnObjs.n_elements

	return R_avg