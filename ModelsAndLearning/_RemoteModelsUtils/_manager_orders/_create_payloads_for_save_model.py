import ujson

def _create_payloads_for_save_model(ports_of_groups):

	payloads = {}
	for group in ports_of_groups.keys():
		payloads[group] = ujson.dumps({"order": "save_model"})

	return payloads