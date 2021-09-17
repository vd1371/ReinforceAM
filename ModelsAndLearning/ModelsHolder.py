from ._LocalModelsUtils import *

class ModelsHolder:

	def __init__(self, should_warm_up, **params):

		self.warm_up = should_warm_up

		self.IDs = params.pop("env").asset_IDs
		self.n_states = params.pop("n_states")
		self.n_trained = params.pop('n_trained')
		self.base_direc = params.pop("base_direc")
		learning_model = params.pop("learning_model")

		self._create_models(learning_model)

	def _create_models(self, learning_model):
		self.models = {}
		for id_ in self.IDs:
			model = learning_model(n_trained = self.n_trained,
									warm_up = self.warm_up,
									n_states = self.n_states,
									asset_id = id_,
									base_direc = self.base_direc)

			self.models[id_] = model

	def predict_actions_for_all(self, S, eps = 0):
		A = {}
		for id_ in S:
			A[id_] = self.models[id_].predict_actions(S[id_],
														eps = eps)
		return A

	def predict_critic_values_for_all(self, S_hist):
		critics = {}
		for id_ in S_hist:
			critics[id_] = self.models[id_].predict_critic_values(S_hist[id_])
		return critics

	def partial_fit_A2C(self, *args, **kwargs):
		partial_fit_local_A2C(self.models, *args, **kwargs)

	def fit(self, buckets, hyps):
		for j, id_ in enumerate(buckets.IDs):
			self.models[id_].fit(buckets[id_], hyps)

	def predict_Q_values_for_id(self, S_at_step, id_):
		return self.models[id_].predict_Q_values(S_at_step)

	def predict_actions_for_id(self, S, eps):
		return self.models[id_].predict_actions(S, eps)


	def partial_fit_DQN(self, *args, **kwargs):
		partial_fit_local_DQN(self.models, *args, **kwargs)

	def update(self, by_other_models_holder = None):
		update_local_models(self.models, by_other_models_holder)

	def save_all(self):
		for id_ in self.IDs:
			n_trained = self.models[id_].save()
		return n_trained