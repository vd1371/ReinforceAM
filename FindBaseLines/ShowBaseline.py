
from .FindBaseLines import find_baselines

def show_baseline(for_, should_find_baselines, LrnObjs, Run):

	# Simulating for the fixed plan
	fixed_plan = find_baselines(LrnObjs.n_assets,
								"GAbyRF",
								should_find_baselines,
								LrnObjs.base_direc)
	# fixed_plan = load_fixed_plan_from(report_direc, "GAbyRF")
	# fixed_plan = optimal_plan()
	R_opt, ac_opt, uc_opt = Run(LrnObjs,
								for_ = "SimOpt",
								fixed_plan = fixed_plan)

	print (f"Opt: R:{R_opt:.2f} | AC:{ac_opt:.2f} | UC:{uc_opt:.2f}")
	print ("Results of fixed plan are calculated")

	return R_opt, ac_opt, uc_opt