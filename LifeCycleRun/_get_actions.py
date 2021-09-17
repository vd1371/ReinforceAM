import numpy as np
from ._get_actions_of_fixed_plan import _get_actions_of_fixed_plan

def get_actions(S = None,
				LrnObjs = None,
				fixed_plan = None,
				step = None,
				for_ = None,
				models_holder = None):

	if fixed_plan is None:
		eps = 0 if "sim" in for_.lower() else LrnObjs.eps
		A = models_holder.predict_actions_for_all(S, eps = eps)

	else:
		A = _get_actions_of_fixed_plan(LrnObjs,
									fixed_plan,
									step)

	return A
			

	