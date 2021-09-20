import ast
import ujson
import numpy as np

def handle_partial_fit_remote_A2C(server, post_data):

	S_hist = post_data['S_hist']
	onehot_encoded_actions = \
			post_data['onehot_encoded_actions']
	advantages = post_data['advantages']
	discounted_rs = post_data['discounted_rs']

	for id_ in server.info['ids']:

		server.models[id_].partial_fit_actor(S_hist[str(id_)],
										np.array(onehot_encoded_actions[str(id_)]),
										np.array(advantages[str(id_)]))

		server.models[id_].partial_fit_critic(S_hist[str(id_)],
										np.array(discounted_rs[str(id_)]))

	return "".encode()