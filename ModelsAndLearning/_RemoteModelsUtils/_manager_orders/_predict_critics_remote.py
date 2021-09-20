from ._make_request import _make_request
from ._share_load import share_load

from ._create_payloads_for_predict_critics import _create_payloads_for_predict_critics
from ._interpret_predict_critics_results import _interpret_predict_critics_results


def predict_critics_remote(ports_of_groups, S_hist, groups_of_ids):

	payloads = _create_payloads_for_predict_critics(ports_of_groups,
													S_hist,
													groups_of_ids)

	results = share_load(ports_of_groups, payloads, groups_of_ids)

	results = _interpret_predict_critics_results(results)
	
	return results

