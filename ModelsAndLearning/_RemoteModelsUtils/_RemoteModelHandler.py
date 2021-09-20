import ujson

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.parse import parse_qs

from ._handlers import *

class RemoteModelHandler(BaseHTTPRequestHandler):

	def __init__(self, *args, **kwargs):

		self.server_object = args[-1]
		super().__init__(*args, **kwargs)

	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()

	def do_POST(self):
		self._set_response()
		# query_components = parse_qs(urlparse(self.path).query)

		content_length = int(self.headers['Content-Length'])
		post_data = self.rfile.read(content_length)
		post_data = ujson.loads(post_data)

		order = post_data['order']
		if order == "save_model":
			response = handle_save_model_order(self.server, post_data)

		elif order == "predict_actions_for_all":
			response = handle_predict_actions_for_all_order(self.server,
															post_data)
		elif order == "predict_critics":
			response = handle_predict_critics_order(self.server, post_data)

		elif order == "partial_fit_A2C":
			response = handle_partial_fit_remote_A2C(self.server, post_data)

		self.wfile.write(response)
		self.wfile.flush()

	def log_message(self, format, *args):
		return

	def do_GET(self):

		content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
		post_data = self.rfile.read(content_length) # <--- Gets the data itself

		self._set_response()

		# recevies some data and trains the model