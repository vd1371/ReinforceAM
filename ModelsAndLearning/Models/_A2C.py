#Loading dependencies
import os
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l1, l2
import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
	from .BaseModel import BaseModel
	from .epsilon_greedy import epsilon_greedy
except ImportError:
	from BaseModel import BaseModel
	from epsilon_greedy import epsilon_greedy

#--------------------------------------------------------#
#                                                        #
# Advantage Actor critic model                           #
#                                                        #
#--------------------------------------------------------#
class A2C(BaseModel):

	name = "A2C"

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

			inputs = Input(shape=(self.n_states,))
			common = Dense(30, activation="relu", kernel_regularizer = l2(0.000001))(inputs)
			common = Dense(30, activation="relu", kernel_regularizer = l2(0.000001))(common)
			action = Dense(self.dim_actions, activation="softmax")(common)
			# critic = Dense(1, activation = 'linear')(common)

			optimizer = keras.optimizers.Adam(lr = 0.001)
			actor_model = keras.Model(inputs = inputs, outputs = action)
			actor_model.compile(loss="categorical_crossentropy", optimizer=optimizer)

			# critic_inputs = Input(shape=(self.n_states,))
			# common = Dense(30, activation="relu")(critic_inputs)
			# common = Dense(30, activation="relu")(common)
			critic = Dense(1, activation = 'linear')(common)

			optimizer = keras.optimizers.Adam(lr = 0.001)
			critic_model = keras.Model(inputs = inputs, outputs = critic)
			critic_model.compile(loss="mse", optimizer=optimizer)

			self.models_for_elements.append({'actor': actor_model, 'critic': critic_model})

	def fit(self, bucket, hyps):
		'''For memory replay'''
		for _ in range(hyps['epochs']):
			states, actions, advantages, rewards = bucket.sample_saar(hyps['batch_size'])

			self.partial_fit_actor(states, actions, advantages)
			self.partial_fit_critic(states, rewards)

	def predict_actions(self, X, *args, **kwargs):

		selected_actions = []
		for ne in range(self.n_elements):
			actions_prob = self.models_for_elements[ne]['actor'](np.atleast_2d(X[ne]), training = False).numpy()[0]
			selected_actions.append(np.random.choice(self.dim_actions, p = actions_prob))

		return selected_actions

	def predict_critic_values(self, X):
		critic_values = []

		for ne in range(self.n_elements):
			critic_value = self.models_for_elements[ne]['critic'](np.atleast_2d(X[ne]), training = False).numpy()
			critic_values.append(np.squeeze(critic_value))

		return critic_values

	def partial_fit_actor(self, X, Y, advantages, *args, **kwargs):
		'''Fitting'''
		for ne in range(self.n_elements):
			x = np.atleast_2d(X[ne])
			y = Y[ne]
			adv = advantages[ne]

			self.models_for_elements[ne]['actor'].fit(x, y, sample_weight = adv, epochs = 1, verbose = 0)

		self.n_trained += 1

	def partial_fit_critic(self, X, Y, *args, **kwargs):
		'''Fitting critic model'''
		for ne in range(self.n_elements):
			x = np.atleast_2d(X[ne])
			y = Y[ne]

			self.models_for_elements[ne]['critic'].fit(x, y, epochs = 1, verbose = 0)

	def save(self):
		for ne in range(self.n_elements):
			self.models_for_elements[ne]['actor'].save(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-Actor.h5")
			self.models_for_elements[ne]['critic'].save(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-Critic.h5")
		
		return self.n_trained

	def load(self):
		self.models_for_elements = []
		for ne in range(self.n_elements):
			actor_model = load_model(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-Actor.h5")
			critic_model = load_model(self.base_direc + "Models/" + f"A{self.asset_id}-E{ne}-Critic.h5")

			self.models_for_elements.append({'actor': actor_model, 'critic': critic_model})

	def set_new_weights_from(self, new_models):
		for ne, element_model in enumerate(new_models.models_for_elements):
			
			actor_weights = element_model['actor'].get_weights()
			self.models_for_elements[ne]['actor'].set_weights(actor_weights)

			critic_weights = element_model['critic'].get_weights()
			self.models_for_elements[ne]['critic'].set_weights(critic_weights)
