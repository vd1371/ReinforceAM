import numpy as np
from RLUtils import discount

def calculate_rewards(LrnObjs):

	S_hist, _, ut_hist, _, ac_hist, uc_hist, _ = LrnObjs.episode_holder.get()

	enough_budget = LrnObjs.env.remaining_npv_budget > 0
	valid_A = LrnObjs.validator.is_valid()

	R_hist = {}

	for id_ in ut_hist.keys():

		rewards = np.zeros((LrnObjs.n_elements, LrnObjs.settings.n_steps))

		for ne in range(LrnObjs.n_elements):

			rewards[ne] = ut_hist[id_][ne] / \
							(np.array(uc_hist[id_][ne]) + \
								np.array(ac_hist[id_][ne]))**0.2

			# discounted_uc = discount(uc_hist[id_][ne], LrnObjs.discount_vec)
			# discounted_ac = discount(ac_hist[id_][ne], LrnObjs.discount_vec)

			# if discounted_uc > 0:
			# 	rewards[ne] = ut_hist[id_][ne]/((discounted_ac +
			# 										discounted_uc) ** 0.2)
	
			# discounted_ut = discount(ut_hist[id_][ne], LrnObjs.discount_vec)
			# discounted_ac = discount(ac_hist[id_][ne], LrnObjs.discount_vec)

			# if discounted_uc > 0:
			# 	R = discounted_ut / discounted_uc ** 0.2
			# else:
			# 	R = 0

			# rewards[ne][-1] = R

		rewards[np.isnan(rewards)] = 0
		R_hist[id_] = rewards

	return R_hist