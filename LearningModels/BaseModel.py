
class BaseModel:
	def __init__(self, **params):
		self.n_trained = params.pop("n_trained", 0)
		self.warm_up = params.pop("warm_up", False)
		self.n_elements = params.pop("n_elements", 3)
		self.dim_actions = params.pop("dim_actions", 4)
		self.n_states = params.pop("n_states", 12)
		self.asset_id = params.pop("asset_id", 1)
		self.base_direc = params.pop("base_direc")

	def fit(self, bucket, hyps):
		'''For memory replay'''
		for _ in range(hyps['epochs']):
			x_sa, y_r = bucket.sample_sr(hyps['batch_size'])
			self.partial_fit(x_sa, y_r, hyps['lr'])