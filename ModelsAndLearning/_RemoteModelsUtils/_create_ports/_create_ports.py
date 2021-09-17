from .check_if_port_open import check_if_port_open

def create_ports(MasterLrnObjs):

	next_port = 20000

	ports_and_ids = {}

	for id_ in MasterLrnObjs.env.asset_IDs:

		port_open = False
		while not port_open:
			port_open = check_if_port_open(next_port)
			next_port += 1

		ports_and_ids[id_] = next_port-1

	return ports_and_ids