def partial_fit_DQN(LrnObjs, states, targets):

	for id_ in LrnObjs.env.asset_IDs:
		# Partailly fitting the models
		LrnObjs.models[id_].partial_fit(states[id_],
										targets[id_],
										lr = LrnObjs.lr)