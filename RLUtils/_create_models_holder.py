from ModelsAndLearning import *

def create_models_holder(n_jobs, warm_up, LrnObjs):

	target_models_holder = None
	if n_jobs == 1:

		models_holder = ModelsHolder(should_warm_up = warm_up,
										**LrnObjs.__dict__)

		if learning_model.name != "A2C":
			target_models_holder = ModelsHolder(should_warm_up = True,
										**LrnObjs.__dict__)

	else:
		models_holder = RemoteModelsManager(LrnObjs)

	return models_holder, target_models_holder

	