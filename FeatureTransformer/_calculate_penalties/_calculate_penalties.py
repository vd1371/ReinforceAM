import numpy as np
from copy import deepcopy

from ._get_p_hist_from_validator import _get_p_hist_from_validator
from ._consider_npv_budget import _consider_npv_budget
from ._consider_annual_budget import _consider_annual_budget
from ._consider_usefulness import _consider_usefulness

def calculate_penalties(LrnObjs):

	S_hist, A_hist, ut_hist, \
			nextS_hist, ac_hist, uc_hist, \
				enough_annual_budget = LrnObjs.episode_holder.get()

	penalty_val = -5

	P_hist = _get_p_hist_from_validator(LrnObjs, penalty_val)
	if LrnObjs.n_assets == 1:
		P_hist = _consider_npv_budget(P_hist, S_hist, LrnObjs, penalty_val)
	else:
		P_hist = _consider_annual_budget(P_hist,
										enough_annual_budget,
										LrnObjs, penalty_val)
	
	P_hist = _consider_usefulness(P_hist, ut_hist, uc_hist,
									A_hist, LrnObjs, penalty_val)

	return P_hist
			


