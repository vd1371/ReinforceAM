from ._make_request import _make_request
from ._share_load import share_load

from ._create_payloads_for_fit_remote_A2C import _create_payloads_for_fit_remote_A2C

def partial_fit_remote_A2C(ports,
							S_hist,
							onehot_encoded_actions,
							advantages,
							discounted_rs):

	payloads = _create_payloads_for_fit_remote_A2C(ports,
													S_hist,
													onehot_encoded_actions,
													advantages,
													discounted_rs)

	results = share_load(ports, payloads)
	
	return results
