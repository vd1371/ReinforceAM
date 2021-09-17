import ast
import ujson

def _interpret_save_model_results(results):

	while not results.empty():
		res = results.get()
		res_id_ = ujson.loads(res)

		n_trained = list(res_id_.values())[0]
		return int(n_trained)