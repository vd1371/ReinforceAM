import subprocess

from ._create_ports import create_ports
from ._kick_off_servers import kick_off_servers

class ModelsManager:

	def __init__(self, MasterLrnObjs):
		self.MasterLrnObjs = MasterLrnObjs

		self.ports = create_ports(self.MasterLrnObjs)
		self.kick_off_models_servers()

	def kick_off_models_servers(self):
		self.servers_processes = kick_off_servers(self.ports,
													**self.MasterLrnObjs.__dict__)

	def predict_actions(self, S, eps):
		raise NotImplementedError

	def predict_critic_values(self, S_hist):
		raise NotImplementedError

	def partial_fit_A2C(self, S_hist,
								onehot_encoded_actions,
								advantages,
								discounted_rs):
		raise NotImplementedError

	def partial_fit_DQN(self, S_hist, targets):
		raise NotImplementedError

	def save_all(self):
		raise NotImplementedError

	def copy_from(self, other_models):
		raise NotImplementedError

	def shutdown_servers(self):
		for id_ in self.servers_processes:
			self.servers_processes[id_].terminate()


