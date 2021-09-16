

def partial_fit_A2C(LrnObjs,
					S_hist,
					onehot_encoded_actions,
					advantages,
					discounted_rs):

	for id_ in LrnObjs.env.asset_IDs:
		LrnObjs.models[id_].partial_fit_actor(S_hist[id_],
												onehot_encoded_actions[id_],
												advantages[id_])
		LrnObjs.models[id_].partial_fit_critic(S_hist[id_],
												discounted_rs[id_])