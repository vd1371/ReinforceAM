import pandas as pd

def _save_averages_holders(for_, **params):
	
	base_direc = params.pop("base_direc")

	df = pd.DataFrame()

	df['AgencyCosts'] = params.get("ac_avg_holder")
	df['UserCosts'] = params.get("uc_avg_holder")

	df.to_csv(base_direc + f"/{for_}-CostsAverages.csv")