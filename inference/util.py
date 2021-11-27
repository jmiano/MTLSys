import torch

class model_info():
	"""docstring for model_info"""
	def __init__(self, name, accuracy, load_latency, inf_latency, tasks, model, prune):
		self.name = name
		self.model= model
		self.accuracy = accuracy
		self.load_latency = load_latency
		self.inf_latency = inf_latency #in ms
		self.tasks = tasks
		self.tasks_count = len(tasks)
		self.pruned = prune
		



def load_all_task_models_info(model_file = "model_score_lookup_multitask.tsv", folder="models/model_variants"):
	models = []
	tasks = ["age", "gender", "ethnicity"]
	with open(folder+ "/" + model_file, "r") as f:
		f.readlines()
		for i in f.readlines():
			i = i.split("\t")
			if (i[0] != "ALL"): break
			tasks_count = 3
			inf_latency = i[2] * 1000
			accuracies = []
			## Update this
			load_latency = 10
			for task in range(1,tasks_count+1):
				accuracies.append(i[task+3])
			prune = i[1]*100
			name = "all_p-"+str(prune)+"_MTLModel.pth"
			model_object = load_model(name, folder)
			model = model_info(name, accuracies, load_latency, inf_latency, tasks, model_object, prune)
			models.append(model)
	return models


def load_one_task_models_info(model_file="model_score_lookup_singletask.tsv", folder="models/model_variants", task_name="age", mtl_model=False):
	models = []
	tasks = [task_name]
	tasks_count =1
	with open(folder+ "/" + model_file, "r") as f:
		f.readlines()
		for i in f.readlines():
			i = i.split("\t")
			if (lower(i[0]) != task_name): break
			inf_latency = i[2]*1000
			accuracies = []
			## Update this
			load_latency = 10
			prune = i[1]*100
			for task in range(1,4):
				if(float(i[tasks+3]) != 0.0):
					accuracies.append(i[task])
			if mtl_model:
				name = task_name+"_p-"+prune+"_MTLModel.pth"
			else:
				name = task_name+"_p-"+prune+"SingleTaskModel.pth"
			model_object = load_model(name, folder)
			model = model_info(name, accuracies, load_latency, inf_latency, tasks, model_object, prune)
			models.append(model)
	return models


def load_model(file_name, folder):
	model = torch.load(folder+"/"+file_name)
	return model


def get_models_table(models):
	task_tables = {}
	for i in range(models[0].tasks_count):	
		table = {}
		accuracy_table = {}
		latencies = set([])
		for model in models: 
			model.latency[i] = round(model.latency, 2)
			latencies.add(model.latency)
		for latency in latencies:
			table[latency] = 0
		for model in models:
			model.latency = round(model.latency, 2)
			if(accuracy_table[model.latency] > model.accuracy[i]):
				accuracy_table[model.latency] = model.accuracy[i]
				table[model.latency] = model

		task_tables[models[0].tasks[i]] = table
	return task_tables


