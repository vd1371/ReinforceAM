import time
import json
import os
from copy import deepcopy

from subprocess import Popen, PIPE

from multiprocessing import Process

from ._find_shared_info_for_servers import find_shared_info

def start_one_server(temp_info):

	p = Popen(['py',
					'-3.8',
					'./ModelsAndLearning/RemoteModelsServer.py'],
					stdin = PIPE,
					shell = False)
	p.stdin.write(json.dumps(temp_info).encode())
	p.stdin.close()

def kick_off_servers(ports, **params):

	print ("About to create servers....")

	shared_info = find_shared_info(**params)

	servers_processes = {}
	for id_, port in ports.items():

		temp_info = deepcopy(shared_info)
		temp_info['id_'] = int(id_)
		temp_info['port'] = ports[id_]

		p = Popen(['py',
					'-3.8',
					'./ModelsAndLearning/RemoteModelsServer.py'],
					stdin = PIPE,
					shell = False)
		p.stdin.write(json.dumps(temp_info).encode())
		p.stdin.close()

		# p = Process(target = start_one_server, args = (temp_info,))
		# p.start()

		servers_processes[id_] = p

	print ("Servers are up now")
	time.sleep(5)

	return servers_processes