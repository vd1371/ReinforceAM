#Loading dependenvies
import os
import sys
import time

import numpy as np

from RLUtils import *
from LearningObjects import LearningObjects
from FindBaseLinesModules import *

def find_baselines(n_assets, for_):
	
	print ("Learning started")
	base_direc, _ = create_path(__file__, for_)

	if for_ == 'GAbyRF':
		GeneticAlgorithmByRF(LearningObjects, base_direc, n_assets = n_assets)
	elif for_ == 'GreedyIUC':
		# GreedyIUC(LearningObjects, base_direc, n_assets = n_assets)
		GreedyIUCParallel(LearningObjects, base_direc, n_assets = n_assets)
	else:
		raise ValueError ("for_ must be either GAbyRF or GreedyIUC. "
							"Check for typo")

if __name__ == "__main__":
	exec(n_assets = 4, for_ = 'GAbyRF')