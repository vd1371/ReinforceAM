

def partial_fit_local_A2C(models,
						S_hist,
						onehot_encoded_actions,
						advantages,
						discounted_rs):

	for id_ in S_hist:
		models[id_].partial_fit_actor(S_hist[id_],
									onehot_encoded_actions[id_],
									advantages[id_])
		models[id_].partial_fit_critic(S_hist[id_],
									discounted_rs[id_])