import json
import ujson
import ast

import sys

from ..bytes_to_numpy import bytes_to_numpy
from ..numpy_to_bytes import numpy_to_bytes

def handle_predict_critics_order(server, query_components):

	S = query_components['S']

	critics = {}
	for id_ in server.info['ids']:
		cr = server.models[id_].predict_critic_values(S[str(id_)])
		critics[id_] = cr.tolist()

	return ujson.dumps(critics).encode()
