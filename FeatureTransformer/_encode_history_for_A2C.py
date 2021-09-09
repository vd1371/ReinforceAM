from ._discount_rewards import discount_rewards
import tensorflow as tf

def _encode_history_for_A2C(S_hist,
							A_hist,
							R_hist,
							models,
							gamma,
							dim_actions):
	'''Encoding the rewards to target values'''
	actions, advantages, discounted_rs = {}, {}, {}
	for id_ in S_hist:
		
		# Compute discounted rewards
		discounted_rs[id_] = discount_rewards(R_hist[id_], gamma)

		# get the critic network predictions
		critics = models[id_].predict_critic_values(S_hist[id_])

		# Compute advantages
		advantages[id_] = (discounted_rs[id_] - critics)

		actions[id_] = tf.one_hot(A_hist[id_], dim_actions).numpy()

	return actions, advantages, discounted_rs