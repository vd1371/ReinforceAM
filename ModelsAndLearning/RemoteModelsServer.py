import ast
import json
import ujson
import os

from http.server import HTTPServer
from functools import partial

from _RemoteModelsUtils import *
from Models import *

learning_models_dict = {
		"DQN": DQN,
		"A2C": A2C,
		"DuelingDQN": DuelingDQN
	}

def create_agent_server(info):

	info = ujson.loads(info)
	port = info['port']

	server_address = ("localhost", port)
	server = HTTPServer(server_address, RemoteModelHandler)

	server = bind_model_to_server(server, info, learning_models_dict)

	server.serve_forever()



if __name__ == "__main__":

	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	info = input()

	create_agent_server(info)








