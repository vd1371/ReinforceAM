def memory_replay(LrnObjs, for_):

	# Memory replay
	if not for_ == 'A2C' and LrnObjs.buckets.is_ready():

		LrnObjs.logger.info ("Starting to do memory replay...")
		for j, id_ in enumerate(LrnObjs.buckets.IDs):
			# Memory replay
			LrnObjs.models[id_].fit(LrnObjs.buckets[id_], LrnObjs.hyps)

		# Empty buckets
		LrnObjs.buckets.throw_away()
		LrnObjs.logger.info("Memory replay done.")
