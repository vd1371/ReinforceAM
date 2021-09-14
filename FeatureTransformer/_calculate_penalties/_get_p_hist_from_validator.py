from copy import deepcopy

def _get_p_hist_from_validator(LrnObjs, penalty_val):

	# The penalties are based on the validation of actions
	P_hist = deepcopy(LrnObjs.validator.A_valid_hist)
	for id_ in P_hist:
		P_hist[id_][P_hist[id_] == 0] = penalty_val
		P_hist[id_][P_hist[id_] == 1] = 0

	return P_hist