import numpy as np
#--------------------------------------------------------#
#                                                        #
# Epsilon Greedy                                         #
#                                                        #
#--------------------------------------------------------#
def epsilon_greedy(selected_actions, eps, n_elements = 3, dim_actions = 4):
	'''Conducting epsilon greedy for generating new actions'''

	# Random actions for elements
	random_actions = np.random.choice(dim_actions, size = n_elements)

	# For selecting between random and selected actions
	binary_eps = np.random.choice([0, 1],
										size = n_elements,
										p = [eps, 1-eps])
	# Conducting epsilon greedy
	actions = selected_actions * binary_eps + \
				random_actions * (1-binary_eps)

	return actions
