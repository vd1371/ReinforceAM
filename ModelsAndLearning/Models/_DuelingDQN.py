#Loading dependencies
import os
import numpy as np

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input, Add, Lambda
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
	from .BaseModel import BaseModel
	from .epsilon_greedy import epsilon_greedy
except ImportError:
	from BaseModel import BaseModel
	from epsilon_greedy import epsilon_greedy

#--------------------------------------------------------#
#                                                        #
# DQN                                                    #
#                                                        #
#--------------------------------------------------------#
class DuelingDQN(BaseModel):
	
	name = "DuelingDQN"

	def __init__(self, **params):
		super().__init__(**params)

		if self.warm_up:
			self.load()

		else:
			self._create_models()
			self.save()

	def _create_models(self):
		'''Creating models'''
		self.models_for_elements = []

		for ne in range(self.n_elements):

			X_input = Input(shape=(self.n_states,))
			X = X_input
			X = Dense(30, input_shape=(self.n_states,),
						activation="tanh",
						kernel_regularizer = l2(0.000001))(X)
			X = Dense(30, activation="relu", kernel_regularizer = l2(0.000001))(X)

			state_value = Dense(1)(X)
			state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
									output_shape=(self.dim_actions,))(state_value)

			action_advantage = Dense(self.dim_actions)(X)
			action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
										output_shape=(self.dim_actions,))(action_advantage)

			X = Add()([state_value, action_advantage])

			model = Model(inputs = X_input, outputs = X)

			optimizer = Adam(lr = 0.001)
			model.compile(loss='mse', optimizer=optimizer)
			
			self.models_for_elements.append(model)

	def partial_fit(self, X, Y, *args, **kwargs):
		'''Fitting'''
		for ne in range(self.n_elements):
			x = np.atleast_2d(X[ne])
			y = np.array(Y[ne])

			self.models_for_elements[ne].fit(x, y, epochs = 1, verbose = 0)

		self.n_trained += 1

	def predict_actions(self, X, eps):
		'''Choosing the action based on higher q-values

		:params: X a disctionry contatining states
		:return: action for each elemetn corresponding to the highest Q-value
		'''
		Q_values, _ = self.predict_Q_values(X)

		selected_actions = np.argmax(Q_values, axis = 1)
		actions  = epsilon_greedy(selected_actions, eps)

		return actions

	def predict_Q_values(self, X):
		'''finding the q-values for each action of each element

		:params: X an array contating states
		:return: a numpy array with self.n_elements
		'''
		Q_values = []
		for ne in range(self.n_elements):
			Q_values.append(self.models_for_elements[ne](np.atleast_2d(X[ne]), training = False).numpy()[0])

		maxQ_values = np.max(Q_values, axis = 1)
		return np.array(Q_values), maxQ_values

	def save(self):
		'''Saving the models'''
		for ne in range(self.n_elements):
			self.models_for_elements[ne].save(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}.h5")
		return self.n_trained

	def load(self):
		'''Loading the models'''
		self.models_for_elements = []
		for ne in range(self.n_elements):
			model = load_model(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}.h5")
			self.models_for_elements.append(model)

	def set_new_weights_from(self, new_models):
		for ne, element_model in enumerate(new_models.models_for_elements):
			
			actor_weights = element_model.get_weights()
			self.models_for_elements[ne].set_weights(actor_weights)