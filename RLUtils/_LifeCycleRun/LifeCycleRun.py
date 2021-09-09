#Loading dependencies
from ._get_actions import get_actions
from ._partial_fit_A2C import partial_fit_A2C
from ._partial_fit_DQN import partial_fit_DQN

def Run(LrnObjs, **params):

	buckets = params.pop('buckets', None)

	for_ = params.pop("for_", "Q")

	valid_models = ['DuelingDQN', 'DQN', 'A2C', 'SGD', 'SGDReg', 'Sim', 'SimOpt']
	if not for_ in valid_models:
		raise ValueError ("for_ must be one of: "
							f"{valid_models}")

	fixed_plan = params.pop("fixed_plan", None)

	N = LrnObjs.n_sim if "Sim" in for_ else 1

	for i in range(N):

		# A new experience
		s_a_rs = LrnObjs.env.reset()
		S, R, ac, uc = LrnObjs.ft.encode_raw_SARS(s_a_rs)

		episode_done = False
		step = 0

		while not episode_done:

			A = get_actions(S = S,
							LrnObjs = LrnObjs,
							step = step,
							fixed_plan = fixed_plan)
		
			# for validation
			LrnObjs.validator.add_to_history(A, s_a_rs[LrnObjs.env.asset_IDs[0]]['step'])

			# Checking ig there is enough budget left, and also valid Action?
			enough_budget = LrnObjs.env.enough_NPV_budget(A)
			# enough_budget = LrnObjs.env.enough_annual_budget(A, step)
			valid_A = LrnObjs.validator.is_valid()

			# Taking the action on a network and finding next s_a_rs
			s_a_rs = LrnObjs.env.step(A)

			# Is it done?
			episode_done = s_a_rs[LrnObjs.env.asset_IDs[0]]['done']
			step = s_a_rs[LrnObjs.env.asset_IDs[0]]['step']

			# Getting the reward and next state, also costs
			nextS, R, ac, uc = \
				LrnObjs.ft.encode_raw_SARS(s_a_rs,
												valid_A,
												enough_budget,
												LrnObjs.n_elements)

			# Updating the episode holder or sim_ana
			LrnObjs.episode_holder.add(S, A, R, ac, uc, nextS)

			S = nextS

		# Getting the states, actions, and rewards of all assets in the life cycle
		S_hist, A_hist, R_hist, nextS_hist, ac, uc = LrnObjs.episode_holder.get()
		LrnObjs.sim_results_holder.update_histogram(A_hist, ac, uc)

		R_avg, ac_avg, uc_avg = LrnObjs.episode_holder.get_episode_results()
		LrnObjs.sim_results_holder.update_averages(R_avg, ac_avg, uc_avg)

		if for_ == "A2C":
			# Encode SARs
			onehot_encoded_actions, advantages, discounted_rs = \
				LrnObjs.ft.encode_history_for_A2C(S_hist,
													A_hist,
													R_hist,
													LrnObjs.models,
													LrnObjs.GAMMA,
													LrnObjs.dim_actions)

			partial_fit_A2C(LrnObjs, S_hist, onehot_encoded_actions,
										advantages, discounted_rs)

			# Adding to buckets for memory replay
			LrnObjs.buckets.add_saar(S_hist, onehot_encoded_actions,
											advantages, discounted_rs)

		elif for_ in ['DuelingDQN', 'DQN', 'SGD', 'SGDReg']:
			# Finding the targets
			targets = LrnObjs.ft.encode_history_for_QLrn(S_hist,
														A_hist,
														R_hist,
														nextS_hist,
														LrnObjs)

			# partial_fit_DQN(LrnObjs, S_hist, targets)

			# Adding to the buckets
			LrnObjs.buckets.add_sr(S_hist, targets)

		# Restarting and refreshing the holders and validator
		LrnObjs.validator.reset_for_each_cycle()
		LrnObjs.episode_holder.reset_for_each_cycle()

	if "sim" in for_.lower():
		LrnObjs.sim_results_holder.get_histogram(for_)

	R_avg, ac_avg, uc_avg = LrnObjs.sim_results_holder.get_avgs()
	LrnObjs.sim_results_holder.refresh()

	return R_avg, ac_avg, uc_avg