from ._make_request import _make_request
from ._share_load import share_load

from ._create_payloads_for_predict_critics import _create_payloads_for_predict_critics
from ._interpret_predict_critics_results import _interpret_predict_critics_results


def predict_critics_remote(ports, S_hist):

	payloads = _create_payloads_for_predict_critics(ports, S_hist)

	results = share_load(ports, payloads)

	results = _interpret_predict_critics_results(results)
	
	return results

