from .check_if_port_open import check_if_port_open

def create_ports(MasterLrnObjs):

	next_port = 20000

	ports_of_groups = {}

	for group in MasterLrnObjs.groups_of_ids:

		port_open = False
		while not port_open:
			port_open = check_if_port_open(next_port)
			next_port += 1

		ports_of_groups[group] = next_port-1

	return ports_of_groups