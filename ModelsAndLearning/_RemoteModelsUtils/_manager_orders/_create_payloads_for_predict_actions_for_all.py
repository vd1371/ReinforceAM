import json
import ujson


def _create_payloads_for_predict_actions_for_all(ports_of_groups,
												S,
												eps,
												groups_of_ids):

	payloads = {}

	for group, ports in ports_of_groups.items():

		S_of_payload = {id_: S[id_] for id_ in groups_of_ids[group]}

		payload = {"order": "predict_actions_for_all",
							"S": S_of_payload,
							"eps": eps}

		payloads[group] = ujson.dumps(payload)

	return payloads

