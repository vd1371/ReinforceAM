import ujson

def _create_payloads_for_predict_critics(ports_of_groups,
										S_hist,
										groups_of_ids):

	payloads = {}

	for group, ports in ports_of_groups.items():
		
		S_of_payload = {id_: S_hist[id_] for id_ in groups_of_ids[group]}

		payload = {"order": "predict_critics",
					"S": S_of_payload}

		payloads[group] = ujson.dumps(payload)

	return payloads