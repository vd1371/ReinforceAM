import json
import ujson
import ast

def handle_predict_actions_for_all_order(server, query_components):

	id_ = server.info['id_']

	S = ujson.loads(query_components['S'][0])
	eps = float(query_components['eps'][0])

	action = server.model.predict_actions(S, eps)

	A = {}
	A[id_] = action

	return ujson.dumps(A).encode()
