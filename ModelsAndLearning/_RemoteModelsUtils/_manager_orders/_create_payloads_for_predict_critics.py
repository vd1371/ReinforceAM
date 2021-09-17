import ujson

def _create_payloads_for_predict_critics(ports, S_hist):

	payloads = {}

	for id_ in ports:
		payload = {"order": "predict_critics",
					"S": ujson.dumps(S_hist[id_])}

		payloads[id_] = payload

	return payloads