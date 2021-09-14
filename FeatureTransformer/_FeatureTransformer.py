from ._encode_raw_SARS_PL import _encode_raw_SARS_PL
from ._encode_raw_SARS_NL_detailed import _encode_raw_SARS_NL_detailed
from ._encode_raw_SARS_NL_summary import _encode_raw_SARS_NL_summary
from ._encode_history_for_A2C import _encode_history_for_A2C
from ._encode_history_for_QLrn import _encode_history_for_QLrn

class FeatureTransformer:

	def __init__(self, **params):
		self.n_assets = params.pop("n_assets")
		self.with_detailed_features = params.pop("with_detailed_features")

	def encode_raw_SARS(self, *args, **kwargs):
		if self.n_assets == 1:
			return _encode_raw_SARS_PL(*args, **kwargs)

		else:
			if self.with_detailed_features:
				return _encode_raw_SARS_NL_detailed(*args, **kwargs)

			else:
				return _encode_raw_SARS_NL_summary(*args, **kwargs)

	def encode_history_for_A2C(self, *args, **kwargs):
		return _encode_history_for_A2C(*args, **kwargs)

	def encode_history_for_QLrn(self, *args, **kwargs):
		return _encode_history_for_QLrn(*args, **kwargs)

