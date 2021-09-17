

def find_n_states(n_assets, with_detailed_features):
	if n_assets == 1 or with_detailed_features:
		n_states = 7*n_assets + 2 # 7*n+2 for detailed,
	else:
		n_states = 49 + 7

	return n_states