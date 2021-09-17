
def bind_model_to_server(server, info, learning_models_dict):

	model_name = info['model_name']
	learning_model = learning_models_dict[model_name]

	server.info = info

	server.model = learning_model(n_trained = info["n_trained"],
					warm_up = info["warm_up"],
					n_states = info["n_states"],
					asset_id = info["id_"],
					base_direc = info["base_direc"])

	return server