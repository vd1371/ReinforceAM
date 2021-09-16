import numpy as np

from ._get_direc_to_data import _get_direc_to_data
from ._load_and_concat_all_csvs import _load_and_concat_all_csvs
from ._covert_binary_plans_to_decimal import _covert_binary_plans_to_decimal
from ._get_plans_from_df_for_asset import _get_plans_from_df_for_asset

def find_all_optimized_actions(**params):

	env = params.get("env")
	IDs = env.asset_IDs

	n_elements = params.get('n_elements')
	n_steps = params.get("settings").n_steps
	dim_actions = params.get("dim_actions")
	n_assets = params.get("n_assets")

	direc = _get_direc_to_data()
	df = _load_and_concat_all_csvs(direc)
	df = _covert_binary_plans_to_decimal(df)
	
	actions = {}
	for id_ in IDs:
		# actions[id_] = np.random.choice(dim_actions, size = (n_elements, n_steps))
		actions[id_] = _get_plans_from_df_for_asset(df, id_, **params)

	print (">>>> All optimized plans are merged into the system...")

	return actions
