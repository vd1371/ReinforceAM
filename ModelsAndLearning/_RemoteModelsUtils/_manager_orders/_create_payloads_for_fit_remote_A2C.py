import ujson

def _create_payloads_for_fit_remote_A2C(ports_of_groups,
										groups_of_ids,
										S_hist,
										onehot_encoded_actions,
										advantages,
										discounted_rs):
	payloads = {}

	for group, ports in ports_of_groups.items():

		S_of_payload = {id_: S_hist[id_] for id_ in groups_of_ids[group]}
		A_of_payload = {id_: onehot_encoded_actions[id_].tolist() for id_ in groups_of_ids[group]}
		advantages_of_payload = {id_: advantages[id_].tolist() for id_ in groups_of_ids[group]}
		rs_of_payload = {id_: discounted_rs[id_].tolist() for id_ in groups_of_ids[group]}

		payload = {"order": "partial_fit_A2C",
					"S_hist": S_of_payload,
					"onehot_encoded_actions": A_of_payload,
					"advantages": advantages_of_payload,
					"discounted_rs": rs_of_payload}

		payloads[group] = ujson.dumps(payload)

	return payloads