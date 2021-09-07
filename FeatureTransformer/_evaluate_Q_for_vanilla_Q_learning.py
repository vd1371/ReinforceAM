def _evaluate_Q_for_vanilla_Q_learning(id_, nextS_at_step, LrnObjs):

	_, maxQ_values_next = LrnObjs.models[id_].predict_Q_values(nextS_at_step)
	Q = maxQ_values_next

	return Q