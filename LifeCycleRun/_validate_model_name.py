

def _validate_model_name(model_name):

	valid_models = ['DuelingDQN', 'DQN', 'A2C', 'SGD', 'SGDReg', 'Sim', 'SimOpt']
	if not model_name in valid_models:
		raise ValueError ("for_ must be one of: "
							f"{valid_models}")
	return