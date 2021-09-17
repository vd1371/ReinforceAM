from ._encoders_for_feature_transformer import *

class FeatureTransformer:

	def __init__(self, **params):
		self.n_assets = params.pop("n_assets")
		self.with_detailed_features = params.pop("with_detailed_features")

	def encode_raw_SARS(self, *args, **kwargs):
		if self.n_assets == 1:
			return encode_raw_SARS_PL(*args, **kwargs)

		else:
			if self.with_detailed_features:
				return encode_raw_SARS_NL_detailed(*args, **kwargs)

			else:
				return encode_raw_SARS_NL_summary(*args, **kwargs)

	def encode_history_for_A2C(self, *args, **kwargs):
		return encode_history_for_A2C(*args, **kwargs)

	def encode_history_for_QLrn(self, *args, **kwargs):
		return encode_history_for_QLrn(*args, **kwargs)

