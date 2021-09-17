import ast
import json
import ujson

def _interpret_predict_actions_for_all_results(results):

	A = {}
	while not results.empty():
		res = results.get()

		A_id_ = ujson.loads(res)
		for id_, action in A_id_.items():
			A[int(id_)] = action

	del results

	return A