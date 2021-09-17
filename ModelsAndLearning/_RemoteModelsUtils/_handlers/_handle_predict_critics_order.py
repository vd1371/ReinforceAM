import json
import ujson
import ast

import sys

from ..bytes_to_numpy import bytes_to_numpy
from ..numpy_to_bytes import numpy_to_bytes

def handle_predict_critics_order(server, query_components):

	id_ = server.info['id_']
	S = ujson.loads(query_components['S'][0])
	cr = server.model.predict_critic_values(S)

	critics = {}
	critics[id_] = cr.tolist()

	return ujson.dumps(critics).encode()
