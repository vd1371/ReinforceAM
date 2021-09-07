#Loading dependenvies
import os
import sys
import time

import numpy as np
from .._MRRHolders import MCMRRHolder
from ._get_one_IUC_instance import get_one_IUC_instance

import multiprocessing as mp
from multiprocessing import Queue, Process

def _greedy_IUC_for_parallel(LearningObjects, base_direc, n_assets, q_out):

	LrnObjs = LearningObjects(base_direc,
								n_assets = n_assets,
								n_sim = 1)

	for i in range (LrnObjs.n_sim):
		actions = get_one_IUC_instance(**LrnObjs.__dict__)
		q_out.put(actions)

def GreedyIUCParallel(LearningObjects, base_direc, n_assets):

	n_cores = mp.cpu_count() - 5
	q_out = Queue()

	pool = []
	start = time.time()
	for _ in range(n_cores):
		worker = Process(target = _greedy_IUC_for_parallel, args = (LearningObjects,
															base_direc,
															n_assets,
															q_out,))
		worker.start()
		pool.append(worker)

	LrnObjs = LearningObjects(base_direc, n_assets = n_assets, n_sim = 100)
	mrr_holders = MCMRRHolder(**LrnObjs.__dict__)

	while any(worker.is_alive() for worker in pool):
		while not q_out.empty():
			mrr_holders.update(q_out.get())

		if mrr_holders.n_samples % 100 == 0 \
			and  mrr_holders.n_samples > 0:
			print (f"{mrr_holders.n_samples} were sampled in {time.time()-start:.2f}")
			start = time.time()
			time.sleep(10)

	for worker in pool:
		worker.join()
	
	mrr_holders.get_baseline()
