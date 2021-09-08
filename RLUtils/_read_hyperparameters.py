import ast

def read_hyperparameters(base_direc):

	with open(base_direc + "Hyps.txt", 'r') as f:
			hyps = ast.literal_eval(f.readlines()[0])

	return hyps