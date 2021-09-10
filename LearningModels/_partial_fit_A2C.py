def partial_fit_A2C(LrnObjs, states, actions, advantages, discounted_rs):

	for id_ in LrnObjs.env.asset_IDs:
		LrnObjs.models[id_].partial_fit_actor(states[id_],
												actions[id_],
												advantages[id_])
		LrnObjs.models[id_].partial_fit_critic(states[id_],
												discounted_rs[id_])

