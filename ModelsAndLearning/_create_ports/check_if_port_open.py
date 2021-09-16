import socket

def check_if_port_open(port_number):

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		try:
			s.bind(("127.0.0.1", port_number))
			port_open = True
		except:
			port_open = False
			
	return port_open

	# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	# 	result_of_check = s.connect_ex(('localhost', port_number))
	# 	print (result_of_check)
	# 	return result_of_check == 0