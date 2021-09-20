import time
import json
import ujson
import os
from copy import deepcopy

from subprocess import Popen, PIPE

from multiprocessing import Process

from ._find_shared_info_for_servers import find_shared_info


def kick_off_servers(ports, **params):

	print ("About to create servers....")

	shared_info = find_shared_info(**params)
	groups_of_ids = params.get("groups_of_ids")

	servers_processes = {}
	for group, port in ports.items():

		server_info = deepcopy(shared_info)
		server_info['group'] = group
		server_info['ids'] = groups_of_ids[group]
		server_info['port'] = port

		p = Popen(['py',
					'-3.8',
					'./ModelsAndLearning/RemoteModelsServer.py'],
					stdin = PIPE,
					shell = False)

		p.stdin.write(ujson.dumps(server_info).encode())
		p.stdin.close()

		servers_processes[group] = p

	time.sleep(len(groups_of_ids))
	print ("Servers are up now")

	return servers_processes