#Loading dependenvies
import os
import sys
import time

import numpy as np

sys.path.append('../GIAMS/')

from RLUtils import *
from LearningObjects import LearningObjects
from FindBaseLinesModules import *

def find_baselines(n_assets = None, for_ = None, should_find_baselines = False):

	if not for_ in ['GAbyRF', 'GreedyIUC']:
		raise ValueError ("for_ must be either GAbyRF or GreedyIUC. "
							"Check for typo")

	base_direc, _ = create_path(__file__, for_)

	if not should_find_baselines:
		actions = load_plans(for_, base_direc)
		return actions

	if for_ == 'GAbyRF':
		actions = GeneticAlgorithmByRF(LearningObjects, base_direc, n_assets = n_assets)
	elif for_ == 'GreedyIUC':
		# GreedyIUC(LearningObjects, base_direc, n_assets = n_assets)
		actions = GreedyIUCParallel(LearningObjects, base_direc, n_assets = n_assets)
		

	return actions

if __name__ == "__main__":
	find_baselines(n_assets = 4, for_ = 'GAbyRF')