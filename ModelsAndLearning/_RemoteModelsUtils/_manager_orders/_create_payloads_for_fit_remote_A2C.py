import ujson

def _create_payloads_for_fit_remote_A2C(ports,
										S_hist,
										onehot_encoded_actions,
										advantages,
										discounted_rs):
	payloads = {}

	for id_ in ports:
		payload = {"order": "partial_fit_A2C",
					"S_hist": ujson.dumps(S_hist[id_]),
					"onehot_encoded_actions": ujson.dumps(onehot_encoded_actions[id_].tolist()),
					"advantages": ujson.dumps(advantages[id_].tolist()),
					"discounted_rs": ujson.dumps(discounted_rs[id_].tolist())}

		payloads[id_] = payload

	return payloads