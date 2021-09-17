import tensorflow as tf

from ._discount_rewards_for_A2C import discount_rewards_for_A2C

def encode_history_for_A2C(LrnObjs, R_hist, P_hist, models_holder):

	S_hist, A_hist, _, _, _, _, _ = LrnObjs.episode_holder.get()

	'''Encoding the rewards to target values'''
	actions, advantages, discounted_rs = {}, {}, {}

	critics = models_holder.predict_critic_values_for_all(S_hist)

	for id_ in S_hist:
		
		# Compute discounted rewards
		discounted_rs[id_] = discount_rewards_for_A2C(R_hist[id_],
														P_hist[id_],
														LrnObjs.GAMMA)

		# Compute advantages
		advantages[id_] = (discounted_rs[id_] - critics[id_])

		actions[id_] = tf.one_hot(A_hist[id_], LrnObjs.dim_actions).numpy()

	return actions, advantages, discounted_rs