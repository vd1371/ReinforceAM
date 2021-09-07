import numpy as np
import pandas as pd

class LearningValsHolder:

	def __init__(self, base_direc):
		self.base_direc = base_direc
		self.refresh()

	def refresh(self):
		self.holder = []

	def keep(self, i, R, ac, uc):
		self.holder.append([i, R, ac, uc])

	def save(self):
		cols = ['Experience', 'Rewards', 'AgencyCosts', 'UserCosts']
		df = pd.DataFrame(self.holder, columns = cols)
		df.to_csv(self.base_direc + "HistoryOfLearning.csv")

