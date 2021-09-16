import pandas as pd
import numpy as np

def _get_plans_from_df_for_asset(df, id_, **params):

	n_elements = params.pop('n_elements')
	n_steps = params.pop("settings").n_steps

	plan = df.loc[id_, 'Elem0-0': f'Elem{n_elements-1}-{n_steps-1}'].values
	plan = np.reshape(plan, (n_elements, n_steps)).astype("int")

	return plan