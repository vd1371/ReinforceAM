# Loading dependencies
import numpy as np
from ._interpret_history import interpret_history

MAINT, REHAB, RECON = 1, 2, 3

class EpisodeHolder:

	def __init__(self, **params):

		self.n_elements = params.pop("n_elements", 3)
		self.n_steps = params.pop('n_steps', 10)
		self.dim_actions = params.pop('dim_actions', 4)
		self.dt = params.pop('dt', 2)
		self.discount_rate = params.pop('discount_rate', 0.03)
		self.logger = params.pop('logger', None)
		self.IDs = params.pop("env").asset_IDs

		self.discount_vec = np.exp(np.arange(0, self.n_steps*self.dt, self.dt) * (-self.discount_rate))

		self.reset_for_each_cycle()

	def reset_for_each_cycle(self):

		self.states, self.rewards, self.actions, self.next_states = {}, {}, {}, {}
		self.agency_costs, self.user_costs = {}, {}

		for id_ in self.IDs:

			self.states[id_] = [[] for _ in range(self.n_elements)]
			self.rewards[id_] = [[] for _ in range(self.n_elements)]
			self.actions[id_] = [[] for _ in range(self.n_elements)]
			self.agency_costs[id_] = [[] for _ in range(self.n_elements)]
			self.user_costs[id_] = [[] for _ in range(self.n_elements)]
			self.next_states[id_] = [[] for _ in range(self.n_elements)]

	def add(self, S, A, R, ac, uc, nextS):

		for id_ in self.IDs:
			for ne in range (self.n_elements):
				self.states[id_][ne].append(S[id_][ne])
				self.actions[id_][ne].append(A[id_][ne])
				self.rewards[id_][ne].append(R[id_][ne])
				self.agency_costs[id_][ne].append(ac[id_])
				self.user_costs[id_][ne].append(uc[id_])
				self.next_states[id_][ne].append(nextS[id_][ne])

	def get_episode_results(self):

		R_avg = interpret_history(self.rewards, self.discount_vec, divide_by_n_elements = self.n_elements)
		ac_avg = interpret_history(self.agency_costs, self.discount_vec)
		uc_avg = interpret_history(self.user_costs, self.discount_vec, divide_by_n_elements = self.n_elements)

		return R_avg, ac_avg, uc_avg


	def get(self):
		return self.states, self.actions, self.rewards, self.next_states, \
						self.agency_costs, self.user_costs
