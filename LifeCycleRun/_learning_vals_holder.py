import numpy as np
import pandas as pd

class LearningValsHolder:

	def __init__(self, base_direc, should_warm_up):
		self.base_direc = base_direc
		self.should_warm_up = should_warm_up
		self.refresh()

	def refresh(self):
		self.holder = []
		self.cols = ['Experience', 'Rewards', 'AgencyCosts', 'UserCosts']
		if self.should_warm_up:
			self.df = pd.read_csv(self.base_direc + "HistoryOfLearning.csv", index_col = 0)
		else:
			self.df = pd.DataFrame(columns = self.cols)


	def keep(self, i, R, ac, uc):
		self.holder.append([i, R, ac, uc])

	def save(self):
		new_df = pd.DataFrame(self.holder, columns = self.cols)
		self.df = self.df.append(new_df, ignore_index = True)
		self.df.reset_index(drop=True, inplace = True)
		self.df.to_csv(self.base_direc + "HistoryOfLearning.csv")

