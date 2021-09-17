

def create_group_of_ids_for_servers(**params):

	ids = params.pop("env").asset_IDs
	n_jobs = params.pop("n_jobs")

	group_of_ids = {}

	segment_length = int(len(ids) / n_jobs)
	if len(ids) % n_jobs != 0:
		segment_length += 1

	for job in range(n_jobs):
		start_idx = job * segment_length
		end_idx = min(len(ids), (job+1)*segment_length)

		group_of_ids[f'group{job}'] = ids[start_idx: end_idx]

	return group_of_ids