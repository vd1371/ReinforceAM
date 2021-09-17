from ._make_request import _make_request
from ._share_load import share_load

from ._create_payloads_for_save_model import _create_payloads_for_save_model
from ._interpret_save_model_results import _interpret_save_model_results

def save_all_remote_models(ports):

	payloads = _create_payloads_for_save_model(ports)

	q_out = share_load(ports, payloads)

	results = _interpret_save_model_results(q_out)
	
	return results