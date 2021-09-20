def find_shared_info(**params):

	n_trained = params.get("n_trained")
	warm_up = params.get("warm_up")
	n_states = params.get("n_states")
	base_direc = params.get("base_direc")
	model_name = params.get("learning_model").name

	shared_info = {"n_trained": n_trained,
					"n_states": n_states,
					"warm_up": warm_up,
					"base_direc": base_direc,
					"model_name": model_name}

	return shared_info