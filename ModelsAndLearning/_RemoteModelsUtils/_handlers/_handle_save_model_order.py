import ujson

def handle_save_model_order(server, query_components):

	id_ = server.info['id_']

	n_trained = server.model.save()

	response = {id_: n_trained}

	return ujson.dumps(response).encode()
