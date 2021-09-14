# Loading dependencies
import numpy as np
from ._discount_and_sum import discount_and_sum

MAINT, REHAB, RECON = 1, 2, 3

class EpisodeHolder:

	def __init__(self, **params):

		self.n_elements = params.get("n_elements", 3)
		self.n_steps = params.get('n_steps', 10)
		self.dim_actions = params.get('dim_actions', 4)
		self.dt = params.get('dt', 2)
		self.discount_rate = params.get('discount_rate', 0.03)
		self.logger = params.get('logger', None)
		self.IDs = params.get("env").asset_IDs
		self.discount_vec = params.get("discount_vec")

		self.reset_for_each_cycle()

	def reset_for_each_cycle(self):

		self.states, self.utilities, self.actions, self.next_states = {}, {}, {}, {}
		self.agency_costs, self.user_costs = {}, {}
		self.enough_annual_budget = []

		for id_ in self.IDs:

			self.states[id_] = [[] for _ in range(self.n_elements)]
			self.utilities[id_] = [[] for _ in range(self.n_elements)]
			self.actions[id_] = [[] for _ in range(self.n_elements)]
			self.agency_costs[id_] = [[] for _ in range(self.n_elements)]
			self.user_costs[id_] = [[] for _ in range(self.n_elements)]
			self.next_states[id_] = [[] for _ in range(self.n_elements)]

	def add(self, S, A, ut, ac, uc, nextS, enough_annual_budget):

		for id_ in self.IDs:
			for ne in range (self.n_elements):
				self.states[id_][ne].append(S[id_][ne])
				self.actions[id_][ne].append(A[id_][ne])
				self.utilities[id_][ne].append(ut[id_][ne])
				self.agency_costs[id_][ne].append(ac[id_][ne])
				self.user_costs[id_][ne].append(uc[id_])
				self.next_states[id_][ne].append(nextS[id_][ne])

		self.enough_annual_budget.append(enough_annual_budget)

	def get_episode_results(self):

		ac_avg = discount_and_sum(self.agency_costs,
									self.discount_vec)
		uc_avg = discount_and_sum(self.user_costs,
									self.discount_vec,
									divide_by_n_elements = self.n_elements)

		return ac_avg, uc_avg


	def get(self):
		return self.states, self.actions, self.utilities, self.next_states, \
						self.agency_costs, self.user_costs, self.enough_annual_budget
