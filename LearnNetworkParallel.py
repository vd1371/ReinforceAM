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
		with_detailed_features = False):

	base_direc, report_direc = create_path(__file__, learning_model.name)
	MasterLrnObjs = LearningObjects(base_direc,
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
						# n_states = 7*n_assets + 2, # 7*n+2 for detailed,
						# 51 features from network + 6 for conds and ages 
						n_states = 49 + 7,
						warm_up = warm_up,
						is_double = is_double,
						with_detailed_features = with_detailed_features)

	# R_opt, ac_opt, uc_opt = show_baseline("GAbyRF", should_find_baselines, LrnObjs, Run)
	
	models_manager = ModelsManager(MasterLrnObjs)

	T = 300
	print (f"After {T} seconds, I will shutdown the servers")
	time.sleep(T)
	models_manager.shutdown_servers()




if __name__ == "__main__":

	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	exec(warm_up = False,
		should_find_baselines = True,
		learning_model = A2C,
		is_double = False,
		n_assets = 2,
		with_detailed_features = False)