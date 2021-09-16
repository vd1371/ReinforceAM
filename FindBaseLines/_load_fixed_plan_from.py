import os
import joblib

def load_fixed_plan_from(report_direc, method):
	plan = joblib.load(report_direc + method + "/FixedPlans.json")
	return plan
