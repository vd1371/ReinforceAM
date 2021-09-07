# Loading dependencies

def save_hyperparameters(base_direc, hyps):
	with open(base_direc + "Hyps.txt", 'w') as f: 
		f.write(str(hyps))

