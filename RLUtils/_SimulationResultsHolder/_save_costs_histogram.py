import pandas as pd

def _save_costs_histogram(**params):

	ac_histogram = params.pop("ac_histogram")
	uc_histogram = params.pop("uc_histogram")
	n_steps = params.pop("n_steps")
	dim_actions = params.pop("dim_actions")
	base_direc = params.pop("base_direc")

	rows_holder_ac = []
	rows_holder_uc = []
	indices = []

	for id_ in ac_histogram.keys():
		for ne, _ in enumerate(ac_histogram[id_]):

			indices.append(f"Asset{id_}-Element{ne}")
			rows_holder_ac.append(ac_histogram[id_][ne])
			rows_holder_uc.append(uc_histogram[id_][ne])

	cols = [f"Step:{i}" for i in range(n_steps)]

	df = pd.DataFrame(rows_holder_ac, index = indices, columns = cols)
	df.to_csv(base_direc + f"/AgencyCosts-Histogram.csv")

	df = pd.DataFrame(rows_holder_uc, index = indices, columns = cols)
	df.to_csv(base_direc + f"/UserCosts-Histogram.csv")


