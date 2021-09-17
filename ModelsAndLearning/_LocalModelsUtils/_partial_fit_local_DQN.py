def partial_fit_local_DQN(models, states, targets):

	for id_ in states:
		# Partailly fitting the models
		models[id_].partial_fit(states[id_],
								targets[id_],
								lr = None)