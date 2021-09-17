import threading
import queue

from ._make_request import _make_request

import time

from multiprocessing import Process, Queue

def share_load(ports, payloads):

	q_out = queue.Queue()
	# q_out = Queue()

	pool = []
	for ide_, port in ports.items():

		payload = payloads[ide_]
		
		worker = threading.Thread(target = _make_request,
									args = (port, payload, q_out,))

		# worker = Process(target = _make_request,
		# 							args = (port, payload, q_out,))

		worker.start()
		pool.append(worker)

	for worker in pool:
		worker.join()

	return q_out