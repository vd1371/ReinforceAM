#Loading dependenvies
import os
import sys
import time
import numpy as np

sys.path.append('../GIAMS/')

from RLUtils import *
from LearningModels import *
from LearningObjects import LearningObjects
from FindBaseLines import find_baselines, optimal_plan
from LifeCycleRun import Run

def exec(warm_up = False,
		learning_model = DQN,
		is_double = False,
		n_assets = 1,
		with_detailed_features = True):
	
	print ("Learning started")
	base_direc, report_direc = create_path(__file__, learning_model.name)
	LrnObjs = LearningObjects(base_direc,
						n_assets,
						learning_model = learning_model,
						Exp = 0,
						max_Exp = 10000,
						GAMMA = 0.97,
						lr = 0.0001,
						batch_size = 1000,
						epochs = 10,
						t = 1,
						eps_decay = 0.001,
						eps = 0.5,
						bucket_size = 10000,
						n_sim = 100,
						n_states = 7*n_assets + 2, # 7*n+2 for detailed,
						# 51 features from network + 6 for conds and ages 
						# n_states = 51 + 6,
						warm_up = warm_up,
						is_double = is_double,
						with_detailed_features = with_detailed_features)

	# Simulating for the fixed plan
	# find_baselines(n_assets, "GAbyRF")
	# fixed_plan = load_fixed_plan_from(report_direc, "GAbyRF")
	fixed_plan = optimal_plan()

	R_opt, ac_opt, uc_opt = Run(LrnObjs,
								for_ = "SimOpt",
								fixed_plan = fixed_plan)

	print (f"Opt: R:{R_opt:.2f} | AC:{ac_opt:.2f} | UC:{uc_opt:.2f}")
	print ("Results of fixed plan are calculated")

	print ("Fixed plan is calculated now")

	# Creating
	start, previous_time = time.time(), time.time()
	for i in range(int(LrnObjs.Exp), int(LrnObjs.max_Exp)):

		Run(LrnObjs, for_ = learning_model.name)
		
		memory_replay(LrnObjs, for_ = learning_model.name)

		LrnObjs.save_models_and_hyperparameters(i, after_each = 100)

		LrnObjs.update_target_models(i,
									after_each = 100,
									for_ = learning_model.name)

		LrnObjs.update_eps(i, after_each = 5)

		previous_time = MonitorLearning(i, LrnObjs, Run,
										R_opt, ac_opt, uc_opt,
										start, previous_time)

		# print (f"Experience {i} is analyzed in {time.time()-previous_time:.2f}")

	print ("Done")



if __name__ == "__main__":
	exec(warm_up = False,
		learning_model = A2C,
		is_double = False,
		n_assets = 1,
		with_detailed_features = True)
