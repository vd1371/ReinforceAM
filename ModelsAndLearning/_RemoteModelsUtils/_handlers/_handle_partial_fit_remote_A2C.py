import ast
import ujson
import numpy as np

def handle_partial_fit_remote_A2C(server, query_components):

	id_ = server.info['id_']
	S_hist = ujson.loads(query_components['S_hist'][0])
	onehot_encoded_actions = \
			ujson.loads(query_components['onehot_encoded_actions'][0])
	advantages = ujson.loads(query_components['advantages'][0])
	discounted_rs = ujson.loads(query_components['discounted_rs'][0])

	onehot_encoded_actions = np.array(onehot_encoded_actions)
	advantages = np.array(advantages)
	discounted_rs = np.array(discounted_rs)


	server.model.partial_fit_actor(S_hist,
									onehot_encoded_actions,
									advantages)
	server.model.partial_fit_critic(S_hist,
									discounted_rs)

	return "".encode()