import subprocess
import time

from ._RemoteModelsUtils import *

class RemoteModelsManager:

	def __init__(self, MasterLrnObjs):
		self.MasterLrnObjs = MasterLrnObjs

		self.ports_of_groups = create_ports(self.MasterLrnObjs)
		self.servers_processes = kick_off_servers(self.ports_of_groups,
												**self.MasterLrnObjs.__dict__)


	def save_all(self):
		n_trained = save_all_remote_models(self.ports_of_groups,
											self.MasterLrnObjs.groups_of_ids)
		return n_trained

	def predict_actions_for_all(self, S, eps = 0):
		A = predict_actions_for_all_remote(self.ports_of_groups,
											S,
											eps,
											self.MasterLrnObjs.groups_of_ids)
		return A

	def predict_critic_values_for_all(self, S_hist):
		critics = predict_critics_remote(self.ports_of_groups,
										S_hist,
										self.MasterLrnObjs.groups_of_ids)
		return critics

	def partial_fit_A2C(self, *args, **kwargs):
		partial_fit_remote_A2C(self.ports_of_groups,
								self.MasterLrnObjs.groups_of_ids,
								*args, **kwargs)

	def predict_Q_values_for_id(self, S_at_step, id_):
		raise NotImplementedError

	def predict_actions_for_id(self, S, eps):
		raise NotImplementedError

	def partial_fit_DQN(self, *args, **kwargs):
		raise NotImplementedError

	def update(self, by_other_models_holder = None):
		raise NotImplementedError

	def shutdown_servers(self):
		for id_ in self.servers_processes:
			self.servers_processes[id_].terminate()


