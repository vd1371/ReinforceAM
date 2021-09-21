#Loading dependenvies
import os
import sys
import time
import numpy as np

sys.path.append('../GIAMS/')

from RLUtils import *
from ModelsAndLearning import *
from FindBaseLines import *
from LifeCycleRun import Run
from LearningObjects import *

import time

def exec(warm_up = False,
		learning_model = DQN,
		should_find_baselines = True,
		is_double = False,
		n_assets = 1,
		with_detailed_features = False,
		n_jobs = 10):
	
	print ("Learning started")
	base_direc, report_direc = create_path(__file__, learning_model.name)
	LrnObjs = LearningObjects(base_direc,
						n_assets,
						learning_model = learning_model,
						Exp = 0,
						max_Exp = 10000,
						GAMMA = 0.97,
						lr = 0.0001,
						batch_size = 100,
						epochs = 10,
						t = 1,
						eps_decay = 0.001,
						eps = 0.5,
						bucket_size = 1000,
						n_sim = 100,
						warm_up = warm_up,
						is_double = is_double,
						with_detailed_features = with_detailed_features,
						n_jobs = n_jobs)

	models_holder, target_models_holder = \
		create_models_holder(n_jobs, warm_up, LrnObjs)

	quit()

	R_opt, ac_opt, uc_opt = \
			show_baseline("GAbyRF", should_find_baselines, LrnObjs, Run)

	start, previous_time = time.time(), time.time()
	for exp in range(int(LrnObjs.Exp), int(LrnObjs.max_Exp)):

		Run(LrnObjs,
			models_holder = models_holder,
			target_models_holder = target_models_holder,
			for_ = learning_model.name)
		
		memory_replay(LrnObjs,
						for_ = learning_model.name,
						models_holder = models_holder)

		checkpoint(exp,
			LrnObjs,
			models_holder,
			target_models_holder,
			for_= learning_model.name,
			checkpoint_freq = 100)

		LrnObjs.update_eps(exp, after_each = 5)

		previous_time = monitor_learning(exp,
										LrnObjs, Run, models_holder,
										R_opt, ac_opt, uc_opt,
										start, previous_time)

		print (f"Experience {exp} is analyzed in {time.time()-previous_time:.2f}")

	models_holder.shutdown_servers()
	print ("Done")



if __name__ == "__main__":

	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"9

	exec(warm_up = False,
		should_find_baselines = True,
		learning_model = A2C,
		is_double = False,
		n_assets = 4700,
		n_jobs = 2,
		with_detailed_features = False)
