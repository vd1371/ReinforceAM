#Loading dependencies
import numpy as np

class Bucket:
	def __init__(self, **params):

		self.n_elements = params.pop("n_elements", 3)
		self.asset_id = params.pop('asset_id', 1)
		self.bucket_size = params.pop("bucket_size", 100)

		self.throw_away()

	def __str__(self):
		return f"Bucket-A{self.asset_id}"

	def __repr__(self):
		return f"Bucket-A{self.asset_id}"
		
	def is_ready(self):
		return len(self.rewards[0]) > self.bucket_size
	
	def throw_away(self):
		self.states = [[] for _ in range(self.n_elements)]
		self.actions = [[] for _ in range(self.n_elements)]
		self.advantages = [[] for _ in range(self.n_elements)]
		self.rewards = [[] for _ in range(self.n_elements)]


	def add_sr(self, state, target):

		for ne in range(self.n_elements):
			self.states[ne] += state[ne]
			self.rewards[ne] += target[ne]

	def add_saar(self, state, action, advantage, reward):
	
		for ne in range(self.n_elements):

			self.states[ne] += state[ne]
			self.actions[ne] += action[ne].tolist()
			self.advantages[ne] += advantage[ne].tolist()
			self.rewards[ne] += reward[ne].tolist()

		
	def sample_sr(self, sample_size=32):
		ne = 0
		choices = np.random.choice(len(self.rewards[ne]), sample_size, replace = False)

		x_sa = [[] for _ in range(self.n_elements)]
		y_r = [[] for _ in range(self.n_elements)]

		for ne in range(self.n_elements):

			x_sa[ne] = np.array(self.states[ne])[choices]
			y_r[ne] = np.array(self.rewards[ne])[choices]
		 
		return x_sa, y_r


	def sample_saar(self, sample_size=32):

		ne = 0
		choices = np.random.choice(len(self.rewards[ne]), sample_size, replace = False)

		states = [[] for _ in range(self.n_elements)]
		actions = [[] for _ in range(self.n_elements)]
		advantages = [[] for _ in range(self.n_elements)]
		rewards = [[] for _ in range(self.n_elements)]

		for ne in range(self.n_elements):

			states[ne] = np.array(self.states[ne])[choices]
			actions[ne] = np.array(self.actions[ne])[choices]
			advantages[ne] = np.array(self.advantages[ne])[choices]
			rewards[ne] = np.array(self.rewards[ne])[choices]

		return states, actions, advantages, rewards



class Buckets:
	def __init__(self, **params):

		self.n_elements = params.pop("n_elements", 3)
		self.bucket_size = params.pop("bucket_size", 100)
		self.IDs = params.pop("env").asset_IDs

		self.throw_away()
		
	def is_ready(self):
		return len(self.buckets[self.IDs[0]].rewards[0]) > self.bucket_size
	
	def throw_away(self):
		self.buckets = {}
		for id_ in self.IDs:
			self.buckets[id_] = Bucket(n_element = self.n_elements,
										bucket_size = self.bucket_size,
										asset_id = id_)
	def add_sr(self, state, target):
		for id_ in self.IDs:
			self.buckets[id_].add_sr(state[id_], target[id_])

	def add_saar(self, state, action, advantage, reward):
		for id_ in self.IDs:
			self.buckets[id_].add_saar(state[id_], action[id_], advantage[id_], reward[id_])
	
	def __getitem__(self, id_):
		return self.buckets[id_]