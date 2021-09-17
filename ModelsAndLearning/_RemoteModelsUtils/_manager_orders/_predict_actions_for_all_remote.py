from ._make_request import _make_request
from ._share_load import share_load

from ._create_payloads_for_predict_actions_for_all import _create_payloads_for_predict_actions_for_all
from ._interpret_predict_actions_for_all_results import _interpret_predict_actions_for_all_results

def predict_actions_for_all_remote(ports, S, eps):

	payloads = _create_payloads_for_predict_actions_for_all(ports, S, eps)

	results = share_load(ports, payloads)

	results = _interpret_predict_actions_for_all_results(results)
	
	return results