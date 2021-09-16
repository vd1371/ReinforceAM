import os
import joblib

def load_plans(for_, base_direc):

	if os.path.exists(base_direc + "FixedPlans.json"):
	 	actions = joblib.load(base_direc + "FixedPlans.json")

	 	return actions
	return None