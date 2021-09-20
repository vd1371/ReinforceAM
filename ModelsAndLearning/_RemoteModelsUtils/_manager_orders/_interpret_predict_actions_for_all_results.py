import ast
import json
import ujson

def _interpret_predict_actions_for_all_results(results_queue):

	A = {}
	while not results_queue.empty():
		res = results_queue.get()

		A_id_ = ujson.loads(res)
		for id_, action in A_id_.items():
			A[int(id_)] = action

	del results_queue

	return A