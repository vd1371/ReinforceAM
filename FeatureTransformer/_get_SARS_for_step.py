import numpy as np

def get_SARS_for_step(S_hist, R_hist, A_hist, nextS_hist, id_, step, n_elements):

	S_at_step = [S_hist[id_][ne][step] for ne in range(n_elements)]
	R_at_step = np.array([R_hist[id_][ne][step] for ne in range(n_elements)])
	A_at_step = [A_hist[id_][ne][step] for ne in range(n_elements)]
	nextS_at_step = [nextS_hist[id_][ne][step] for ne in range(n_elements)]

	return S_at_step, R_at_step, A_at_step, nextS_at_step