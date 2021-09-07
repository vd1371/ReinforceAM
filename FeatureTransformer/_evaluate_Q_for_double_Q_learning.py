import numpy as np

def _evaluate_Q_for_double_Q_learning(id_, nextS_at_step, LrnObjs):

	target_next_by_target_model, _ = \
			LrnObjs.target_models[id_].predict_Q_values(nextS_at_step)

	next_actions = LrnObjs.models[id_].predict_actions(nextS_at_step, eps = 0)

	return target_next_by_target_model[np.arange(LrnObjs.n_elements), next_actions]