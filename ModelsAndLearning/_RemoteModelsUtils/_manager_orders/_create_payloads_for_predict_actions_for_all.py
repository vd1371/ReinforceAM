import json
import ujson


def _create_payloads_for_predict_actions_for_all(ports, S, eps):

	payloads = {}

	for id_ in ports:
		payload = {"order": "predict_actions_for_all",
							# "S": str(S[id_]),
							"S": ujson.dumps(S[id_]),
							"eps": eps}

		payloads[id_] = payload

	return payloads

