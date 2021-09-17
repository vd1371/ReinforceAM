import ast
import json
import ujson
import numpy as np

from ..bytes_to_numpy import bytes_to_numpy
from ..numpy_to_bytes import numpy_to_bytes

def _interpret_predict_critics_results(results):

	critics = {}
	# for res in results:

	while not results.empty():
		res = results.get()
		cr_id_ = ujson.loads(res)
		for id_, cr in cr_id_.items():
			critics[int(id_)] = np.array(cr)

	return critics