#Loading dependencies
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import pickle

from .BaseModel import BaseModel
from .epsilon_greedy import epsilon_greedy

#--------------------------------------------------------#
#                                                        #
# Sklearn gradient decent                                #
#                                                        #
#--------------------------------------------------------#
class SklearnSGD(BaseModel):

	name = "SklearnSGD"

	def __init__(self, **params):
		super().__init__(**params)
		self.fitted = False

		if self.warm_up:
			self.load()

		else:
			self._create_models()
			self.save()

	def _create_models(self):
		'''Creating models'''
		self.models = [[] for _ in range (self.n_elements)]

		for ne in range(self.n_elements):
			for a in range(self.dim_actions):
				self.models[ne].append(SGDRegressor(loss='squared_loss',
													penalty='l2',
													verbose=0,
													learning_rate='invscaling'))
		self.fitted = False
		

	def partial_fit(self, X, Y, *args, **kwargs):
		'''Fitting'''
		self.fitted = True
		for ne in range(self.n_elements):
			x = np.atleast_2d(X[ne])

			for a in range(self.dim_actions):
				y = Y[ne][:, a]
				self.models[ne][a].partial_fit(x, y)

		self.n_trained += 1

	def predict_actions(self, X, eps):
		'''Choosing the action based on higher q-values

		:params: X a disctionry contatining states
		:return: action for each elemetn corresponding to the highest Q-value
		'''

		# Actions without epsilon greedy
		# Corresponding to the highest value is selected
		if not self.fitted:
			actions = np.random.choice([0, 1, 2, 3], size = 3)

		else:
			Q_values, _ = self.predict_Q_values(X)

			selected_actions = np.argmax(Q_values, axis = 1)
			actions  = epsilon_greedy(selected_actions, eps)

		return actions

	def predict_Q_values(self, X):
		'''finding the q-values for each action of each element

		:params: X an array contating states
		:return: a numpy array with self.n_elements
		'''
		if not self.fitted:
			Q_values = np.random.randn(self.n_elements, self.dim_actions)

		else:
			Q_values = [[] for _ in range(self.n_elements)]
			for ne in range(self.n_elements):
				for a in range(self.dim_actions):
					value = self.models[ne][a].predict(np.atleast_2d(X[ne]))[0]
					Q_values[ne].append(value)
		
		maxQ_values = np.max(Q_values, axis = 1)
		return np.array(Q_values), maxQ_values

	def save(self):
		'''Saving the models'''
		for ne in range(self.n_elements):
			for a in range(self.dim_actions):
				with open(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-a{a}.pkl", 'wb') as file:
					pickle.dump(self.models[ne][a], file)
		return self.n_trained

	def load(self):
		'''Loading the models'''
		self.models = [[] for _ in range (self.n_elements)]

		for ne in range(self.n_elements):
			for a in range(self.dim_actions):
				with open(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-a{a}.pkl", 'rb') as file:
					model = pickle.load(file)
					self.models[ne].append(model)
		try:
			check_is_fitted(self.models[0][0], "coef_")
			self.fitted = True

		except NotFittedError:
			self.fitted = False