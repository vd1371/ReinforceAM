# Loading dependencies
import numpy as np
from collections import Counter


MAINT, REHAB, RECON = 1, 2, 3

#---------------------------------------------------#
#
# Actions validator                                 #
#
#---------------------------------------------------#
class ActionsValidator:

	def __init__(self, **params):
		'''ActionsHolder
		
		It shall be used to validate the 
		'''
		self.n_elements = params.pop('n_elements', 3)
		self.n_steps = params.pop('n_steps', 10)
		self.logger = params.pop('logger', None)
		self.IDs = params.pop("env").asset_IDs

		self.reset_for_each_cycle()

	def reset_for_each_cycle(self):
		self.A_hist_assets = {id_: np.zeros((self.n_elements, self.n_steps)) for id_ in self.IDs}

	def add_to_history(self, A, step):

		for id_ in A:
			for elem in range(self.n_elements):
				self.A_hist_assets[id_][elem][step] = A[id_][elem]

	def is_valid(self):

		validation = {}
		for id_ in self.A_hist_assets:
			validation[id_] = self.is_mrr_valid(self.A_hist_assets[id_])

		return validation

	def is_mrr_valid(self, mrr_decimal):

		elements_validation = []

		for elem, elem_mrr in enumerate(mrr_decimal):
			
			valid = True
			counts = Counter(elem_mrr)
			if counts[RECON] > 2:
				valid = False
			elif counts[REHAB] > 3:
				valid = False
			elif counts[MAINT] > 5:
				valid = False

			else:
				valid = not (any(np.array(elem_mrr[:-1]) * np.array(elem_mrr[1:])) > 0)

			elements_validation.append(valid)

		print (mrr_decimal)
		print (elements_validation)

		return elements_validation