import sys
import pprint

from RLUtils import *
from FeatureTransformer import FeatureTransformer
from LearningModels import *

from RLEnvs import IndianaEnv

class LearningObjects:

	def __init__(self, base_direc, n_assets, **params):

		self.warm_up = params.pop("warm_up", False)

		self.base_direc = base_direc
		self.n_assets = n_assets

		self.learning_model = params.pop("learning_model", None)

		if self.warm_up:
			# Hyperparameters
			hyps = read_hyperparameters(self.base_direc)

			for k, v in hyps.items():
				setattr(self, k, v)

		else:
			# Hyperparameters
			self.Exp = params.pop("Exp", 0)
			self.max_Exp = params.pop("max_Exp", 100000)
			self.GAMMA = params.pop("GAMMA", 0.97)
			self.lr = params.pop("lr", 0.0001)
			self.batch_size = params.pop("batch_size", 256)
			self.epochs = params.pop("epochs", 40)
			self.t = params.pop("t", 1)
			self.eps_decay = params.pop("eps_decay", 0.001)
			self.eps = params.pop('eps', 0.5)
			self.bucket_size = params.pop('bucket_size', 5000)
			self.n_sim = params.pop("n_sim", 1000)
			self.n_trained = params.pop("n_trained", 0)
			self.is_double = params.pop("is_double", False)
			self.n_states = params.pop("n_states", 9)
			self.n_elements = params.pop("n_elements", 3)
			self.dim_actions = params.pop("dim_actions", 4)
			self.with_detailed_features = params.pop("with_detailed_features", True)

			self.save_hyperparameters()
			self._log_hyperparameters()

		self._construct()
		

	def save_hyperparameters(self):

		self.hyps = {'Exp': self.Exp,
					'max_Exp' : self.max_Exp,
					'GAMMA' : self.GAMMA,
					'lr': self.lr, #learning rate
					'batch_size' : self.batch_size,
					'epochs' : self.epochs,
					't': self.t,
					'eps_decay' : self.eps_decay,
					'eps' : self.eps,
					'n_trained' : self.n_trained,
					'bucket_size': self.bucket_size,
					'n_sim': self.n_sim,
					'is_double': self.is_double,
					'with_detailed_features': self.with_detailed_features,
					'n_states': self.n_states,
					'n_elements': self.n_elements,

					}
		save_hyperparameters(self.base_direc, self.hyps)

	def _construct(self):
		
		self.logger = Logger(address = self.base_direc + "Log.log")
		self.settings = GeneralSettings()
		self.env = IndianaEnv(**self.__dict__)

		if not self.learning_model is None:
			self.ft = FeatureTransformer(**self.__dict__)
			self.validator = ActionsValidator(**self.__dict__)
			self.models = ModelsHolder(should_warm_up = self.warm_up, **self.__dict__)
			self.target_models = ModelsHolder(should_warm_up = True, **self.__dict__)
			self.buckets = Buckets(**self.__dict__)
			self.episode_holder = EpisodeHolder(**self.__dict__)
			self.learning_vals_holder = LearningValsHolder(self.base_direc)
			self.sim_results_holder = SimResultsHolder(**self.__dict__)

	def _log_hyperparameters(self):
		self.logger.info(pprint.pformat(self.hyps))

	def update_eps(self, experience, after_each):

		if experience > 0 and experience % after_each == 0:
			self.t += self.eps_decay
			self.eps = 0.5 / (self.t + self.eps_decay)

	def update_target_models(self, experience = 0,
									after_each = 100,
									for_ = "DQN"):
		if experience % after_each == 0:
				self.target_models.copy_from(self.models)

	def update_hyperparameters(self, **params):

		for k, v in params.items():
			if hasattr(self, k):
				setattr(self, k, v)
			else:
				raise ValueError ("a key has a typo and does not exists among "
									"the class attributes")

	def save_models_and_hyperparameters(self, experience, after_each):
		if experience % after_each == 0:

			n_trained = self.models.save_all()

			self.save_hyperparameters()

			self.update_hyperparameters(n_trained = n_trained,
											Exp = experience)



