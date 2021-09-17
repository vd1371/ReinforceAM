import numpy as np

def _evaluate_Q_for_Q_learning(is_double,
								id_,
								nextS_at_step,
								models_holder,
								target_models_holder):

	targets_by_target_network, _ = \
		target_models_holder.predict_Q_values_for_id(nextS_at_step, id_)

	target_by_online_model, _ = \
		models_holder.predict_Q_values_for_id(nextS_at_step, id_)

	if is_double:
		# Getting it from the model means one more time matrix multiplications
		action = np.argmax(target_by_online_model, axis = 1)
		Q = targets_by_target_network[action]
	else:
		Q = np.max(targets_by_target_network, axis = 1)
	
	return target_by_online_model, Q