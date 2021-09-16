import json

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.parse import parse_qs

from functools import partial

class ModelHandler(BaseHTTPRequestHandler):

	def __init__(self, *args, **kwargs):

		self.server_object = args[-1]
		super().__init__(*args, **kwargs)

	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()

	def do_GET(self):
		self._set_response()

		try:
			query_components = parse_qs(urlparse(self.path).query)
			# print ("query", urlparse(self.path).query)
			# print ("query_components", query_components

			self.server_object.id_ *= 2

			print ("Now the server_object id_ is", self.server_object.id_)


			self.wfile.write(b"Hello")

		except:
			self.wfile.write(b"NotFound")

		## gets some state and return actions

	def do_POST(self):

		content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
		post_data = self.rfile.read(content_length) # <--- Gets the data itself

		self._set_response()

		# recevies some data and trains the model