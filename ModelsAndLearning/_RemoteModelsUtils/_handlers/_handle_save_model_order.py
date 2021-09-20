import ujson

def handle_save_model_order(server, query_components):

	for id_ in server.info['ids']:

		n_trained = server.models[id_].save()

	return str(n_trained).encode()
