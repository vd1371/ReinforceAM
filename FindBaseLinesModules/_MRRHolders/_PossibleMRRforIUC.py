import numpy as np
smallest_number = np.finfo(np.float64).tiny

class PossibleMRRforIUC:

	def __init__(self, ID, mrr, ac_costs, uc_costs, utils, step):

		self.ID = ID
		self.mrr = mrr

		self.ac_costs = np.sum(ac_costs)
		self.uc_costs = uc_costs
		self.utils = np.sum(utils)
		self.step = step

		self.u_c = self.utils / ((self.ac_costs + self.uc_costs)**0.2 + smallest_number)

	def __repr__(self):
		return f"Asset: {self.ID} - mrr: {self.mrr} - AC: {self.ac_costs} u_c: {self.u_c}"