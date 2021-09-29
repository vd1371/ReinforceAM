import requests

def _make_request(port, payload, q_out):
	
	addr = f"http://127.0.0.1:{port}"
	# results = requests.get(addr, params=payload).content.decode("utf-8")
	response = requests.post(addr, data=payload)
	results = response.content.decode("utf-8")

	response.close()
	response = None

	q_out.put(results)