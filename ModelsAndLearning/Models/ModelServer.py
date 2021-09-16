import ast
import json

from http.server import HTTPServer
from functools import partial

from _A2C import A2C
from _ModelHandler import ModelHandler

from _bind_model_to_server import bind_model_to_server

def create_agent_server(info):

	info = json.loads(info)
	port = info['port']

	server_address = ("localhost", port)
	server = HTTPServer(server_address, ModelHandler)

	server = bind_model_to_server(server, info)

	server.serve_forever()



if __name__ == "__main__":

	info = input()

	create_agent_server(info)








