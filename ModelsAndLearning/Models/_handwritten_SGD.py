#Loading dependencies
import os
import numpy as np
import pandas as pd

try:
	from .BaseModel import BaseModel
	from .epsilon_greedy import epsilon_greedy
except ImportError:
	from BaseModel import BaseModel
	from epsilon_greedy import epsilon_greedy

#--------------------------------------------------------#
#                                                        #
# Hand writtedn gradient decent                          #
#                                                        #
#--------------------------------------------------------#
class HandWrittenSGD(BaseModel):

	name = "HandWrittenSGD"

	def __init__(self, **params):
		super().__init__(**params)

		if self.warm_up:
			self.load()

		else:
			self._create_models()
			self.save()

	def _create_models(self):
		'''Creating models for the first time'''
		# self.models = np.random.choice([0, 1, -1], size = (self.n_elements, self.dim_actions, self.n_states))
		self.models = np.random.randn(self.n_elements, self.dim_actions, self.n_states) / \
							np.sqrt(self.n_states)

	def partial_fit(self, X, Y, lr, *args, **kwargs):
		'''Learning one step with gradient decent

		:params: X and Y are dictionaries consisting observations
		and target values
		:return: nothing
		'''
		for ne in range(self.n_elements):
			x = np.atleast_2d(X[ne])
			for a in range(self.dim_actions):

				y = np.array(Y[ne])[:, a]
				self.models[ne][a] += lr * np.dot((y - np.dot(x, self.models[ne][a])/x.shape[0]), x)
			
		self.n_trained += 1

	def predict_actions(self, X, eps):
		'''Choosing the action based on higher q-values

		:params: X a disctionry contatining states
		:return: action for each elemetn corresponding to the highest Q-value
		'''

		# Actions without epsilon greedy
		# In the following line, the state of each element is dot multiplied into
		# the matrix of weights (self.n_states, self.dim_actions). Then, the action
		# Corresponding to the highest value is selected
		models = np.transpose(self.models, (0, 2, 1))

		selected_actions = np.argmax(np.einsum('ij,ijl->il', X, models), axis = 1)
		actions  = epsilon_greedy(selected_actions, eps)

		return actions

	def predict_Q_values(self, X):
		'''finding the q-values for each action of each element

		:params: X an array contating states
		:return: a numpy array with self.n_elements
		'''
		# Actions without epsilon greedy
		# In the following line, the state of each element is dot multiplied into
		# the matrix of weights (self.n_states, self.dim_actions). Then, the action
		# Corresponding to the highest value is selected
		models = np.transpose(self.models, (0, 2, 1))
		Q_values = np.einsum('ij,ijl->il', X, models)
		maxQ_values = np.max(Q_values, axis = 1)

		return np.array(Q_values), maxQ_values

	def save(self):
		'''Saving the models coeffs'''
		cols = [f"X{i}" for i in range (self.n_states)]

		row_list, idx = [], []
		for ne in range(self.n_elements):
			for ai in range (self.dim_actions):
				idx.append(f"E{ne}-Ac{ai}")

		df = pd.DataFrame(self.models.reshape(self.dim_actions*self.n_elements, -1),
								columns = cols, index = idx)
		df.to_csv(self.base_direc + f'A{self.asset_id}.csv')
		return self.n_trained

	def load(self):
		'''Loading the weights from the saved df'''
		df = pd.read_csv(self.base_direc + f'A{self.asset_id}.csv', index_col = 0)
		self.models = df.values.reshape(self.n_elements, self.dim_actions, self.n_states)