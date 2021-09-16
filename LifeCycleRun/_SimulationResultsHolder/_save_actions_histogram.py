import pandas as pd

def _save_actions_histogram(**params):

	A_histogram = params.pop("A_histogram")
	n_steps = params.pop("n_steps")
	dim_actions = params.pop("dim_actions")
	base_direc = params.pop("base_direc")

	rows_holder = []
	indices = []

	for id_ in A_histogram.keys():
		for ne, element_actions in enumerate(A_histogram[id_]):
			for act in range(dim_actions):

				indices.append(f"Asset{id_}-Element{ne}-Action{act}")
				rows_holder.append(A_histogram[id_][ne][:, act])

	cols = [f"Step:{i}" for i in range(n_steps)]

	df = pd.DataFrame(rows_holder, index = indices, columns = cols)
	df.to_csv(base_direc + "/ActionsHistogram.csv")


