import json
from copy import deepcopy

from subprocess import Popen, PIPE

from ._find_shared_info_for_servers import find_shared_info



def kick_off_servers(ports, **params):

	print ("About to create servers....")

	shared_info = find_shared_info(**params)

	servers_processes = {}
	for id_, port in ports.items():

		temp_info = deepcopy(shared_info)
		temp_info['id_'] = int(id_)
		temp_info['port'] = ports[id_]

		p = Popen(['py', '-3.8', './ModelsAndLearning/Models/ModelServer.py'],
					stdin = PIPE)
		p.stdin.write(json.dumps(temp_info).encode())
		p.stdin.close()

		servers_processes[id_] = p

	print ("Servers are up now")

	return servers_processes