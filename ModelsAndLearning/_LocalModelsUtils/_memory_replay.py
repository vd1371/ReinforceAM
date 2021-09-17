def memory_replay(LrnObjs, for_, models_holder):

	# Memory replay
	if not for_ == 'A2C' and LrnObjs.buckets.is_ready():

		LrnObjs.logger.info ("Starting to do memory replay...")
		
		models_holder.fit(LrnObjs.buckets, LrnObjs.hyps)

		# Empty buckets
		LrnObjs.buckets.throw_away()
		LrnObjs.logger.info("Memory replay done.")
