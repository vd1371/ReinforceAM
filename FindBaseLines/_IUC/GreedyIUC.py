#Loading dependenvies
import os
import sys
import time

import numpy as np
from .._MRRHolders import MCMRRHolder
from ._get_one_IUC_instance import get_one_IUC_instance
from copy import deepcopy

def GreedyIUC(LearningObjects, base_direc, n_assets):

	LrnObjs = LearningObjects(base_direc,
								n_assets = n_assets,
								n_sim = 5)

	mrr_holders = MCMRRHolder(**LrnObjs.__dict__)

	start = time.time()
	for i in range (LrnObjs.n_sim):

		print (f"{i}/{LrnObjs.n_sim} is simulated in {time.time()-start:.2f}")
		start = time.time()
		
		actions = get_one_IUC_instance(**LrnObjs.__dict__)

		mrr_holders.update(actions)

	mrr_holders.get_baseline()

