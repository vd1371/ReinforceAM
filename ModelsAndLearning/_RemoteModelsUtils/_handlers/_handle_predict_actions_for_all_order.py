import json
import ujson
import ast

def handle_predict_actions_for_all_order(server, post_data):

	S = post_data['S']
	eps = float(post_data['eps'])

	A = {}
	for id_ in server.info['ids']:
		action = server.models[id_].predict_actions(S[str(id_)], eps)
		A[id_] = action

	return ujson.dumps(A).encode()
