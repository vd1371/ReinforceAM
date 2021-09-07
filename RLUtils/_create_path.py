import os

def create_path(file, for_):

	base_direc = os.path.join(os.path.dirname(file), f'reports//{for_}//')
	report_direc = os.path.join(os.path.dirname(file), f'reports//')
	model_direc = os.path.join(os.path.dirname(file), f'reports//{for_}//Models')

	if not os.path.exists(base_direc):
		os.mkdir(base_direc)

	if not os.path.exists(model_direc):
		os.mkdir(model_direc)

	return base_direc, report_direc


