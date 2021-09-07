import time

def MonitorLearning(i, LrnObjs, Run, R_opt, ac_opt, uc_opt, start, previus_time, after_each = 100):

	if i > 0 and i % after_each == 0:

		R_avg_mdl, ac_avg_mdl, uc_avg_mdl = Run(LrnObjs, for_ = 'Sim')
				
		LrnObjs.logger.info (f"Experience {i} | "
					# f"Sim: R:{R_avg_sim:.2f}, AC:{ac_avg_sim:.2f}, UC:{uc_avg_sim:.2f} | "
					f"Mdl: R:{R_avg_mdl:.2f}, AC:{ac_avg_mdl:.2f}, UC:{uc_avg_mdl:.2f} | "
					f"Opt: R:{R_opt:.2f}, AC:{ac_opt:.2f}, UC:{uc_opt:.2f} | "
					f"In {time.time() - previus_time:.2f} secs | "
					f"Total time: {time.time()-start:.2f} | ")

		LrnObjs.learning_vals_holder.keep(i, R_avg_mdl, ac_avg_mdl, uc_avg_mdl)
		LrnObjs.learning_vals_holder.save()

		previus_time = time.time()

	return previus_time