import json

def _create_payloads_for_save_model(ports):

	payloads = {}
	for id_ in ports.keys():
		payloads[id_] = {"order": "save_model"}

	return payloads