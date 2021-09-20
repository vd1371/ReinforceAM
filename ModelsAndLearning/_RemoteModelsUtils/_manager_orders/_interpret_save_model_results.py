import ast
import ujson

def _interpret_save_model_results(results):

	while not results.empty():
		res = results.get()
		n_trained = int(res)

		return n_trained