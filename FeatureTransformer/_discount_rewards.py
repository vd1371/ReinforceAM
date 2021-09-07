import numpy as np
# Smallest number such that 1.0 + eps != 1.0
smallest_number = np.finfo(np.float32).eps.item()

def discount_rewards(rewards, gamma):

	discounted_r = np.zeros_like(rewards)
	for ne in range(len(rewards)):

		running_add = 0
		for i in reversed(range(len(rewards[ne]))):

			running_add = running_add * gamma + rewards[ne][i]
			discounted_r[ne][i] = running_add

		discounted_r[ne] -= np.mean(discounted_r[ne]) # normalizing the result
		discounted_r[ne] /= (np.std(discounted_r[ne])+smallest_number) # divide by standard deviation

	return discounted_r
