import tensorflow as tf

from ._discount_rewards_for_A2C import discount_rewards_for_A2C

def _encode_history_for_A2C(S_hist, A_hist, R_hist, P_hist, LrnObjs):

	'''Encoding the rewards to target values'''
	actions, advantages, discounted_rs = {}, {}, {}

	for id_ in S_hist:
		
		# Compute discounted rewards
		discounted_rs[id_] = discount_rewards_for_A2C(R_hist[id_],
														P_hist[id_],
														LrnObjs.GAMMA)

		# get the critic network predictions
		critics = LrnObjs.models[id_].predict_critic_values(S_hist[id_])

		# Compute advantages
		advantages[id_] = (discounted_rs[id_] - critics)

		actions[id_] = tf.one_hot(A_hist[id_], LrnObjs.dim_actions).numpy()

	return actions, advantages, discounted_rs