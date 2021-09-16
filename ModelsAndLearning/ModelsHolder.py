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

	def save_all(self):
		# Save models
		for id_ in self.IDs:
			n_trained = self.models[id_].save()

		return n_trained

	def copy_from(self, other_models):
		for id_ in self.IDs:
			self.models[id_].set_new_weights_from(other_models[id_])

	def __getitem__(self, id_):
		return self.models[id_]


