

def update_local_models(models, by_other_models_holder):

	for id_ in by_other_models_holder.IDs:
		models[id_].set_new_weights_from(by_other_models_holder.models[id_])