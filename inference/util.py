import torch
from models.models import MTLClassifier, AgeRegressor, GenderClassifier, EthnicityClassifier
import time, csv
import numpy as np

class model_info():
	"""docstring for model_info"""
	def __init__(self, name, accuracy, load_latency, inf_latency, tasks, file_path, prune):
		self.name = name
		self.file_path= file_path
		self.accuracy = accuracy
		self.load_latency = load_latency
		self.latency = inf_latency #in ms
		self.tasks = tasks
		self.tasks_count = len(tasks)
		self.pruned = prune
		



def load_all_task_models_info(dataloader, model_file = "model_score_lookup_multitask.tsv", folder="models/model_variants"):
	models = []
	tasks = ["age", "gender", "ethnicity"]
	with open(folder+ "/" + model_file, "r") as f:
		
		file = f.readlines()
		file = file[1:]
		for i in file:
			i = i.split("\t")
			## Remove \n from last item
			i[-1] = i[-1][:-1]
			if (i[0] != "ALL"): break
			tasks_count = 3
			inf_latency = int(float(i[2]) * 1000)
			accuracies = []
			## Update this
			load_latency = 10
			for task in range(1,tasks_count+1):
				accuracies.append(i[task+3])
			prune = int(float(i[1])*100)
			name = "all_p-"+str(prune)+"_MTLModel.pth"
			file_path = folder+"/"+name
			# model_object = load_model(name, folder)
			mean_lat, std_lat = get_lat(file_path, dataloader)
			inf_latency = mean_lat*1000
			model = model_info(name, accuracies, load_latency, inf_latency, tasks, file_path, prune)
			models.append(model)
			row = [i[0], i[1], mean_lat, std_lat, i[4], i[5], i[6]]
			with open("calibareted_," + model_file, 'a', newline='') as f:
			    writer = csv.writer(f, delimiter='\t')
			    writer.writerow(row)

	return models


def load_one_task_models_info(dataloader, model_file="model_score_lookup_singletask.tsv", folder="models/model_variants", task_name="age", mtl_model=False):
	models = []
	tasks = [task_name]
	tasks_count =1
	with open(folder+ "/" + model_file, "r") as f:
		file = f.readlines()
		file = file[1:]
		for i in file:
			i = i.split("\t")
			## Remove \n from last item
			i[-1] = i[-1][:-1]
			if (i[0].lower() != task_name): continue
			inf_latency = int(float(i[2]) * 1000)
			
			accuracies = []
			## Update this
			load_latency = 10
			prune = int(float(i[1])*100)

			for task in range(1,4):
				if(float(i[task+3]) != 0.0):
					accuracies.append(float(i[task+3]))
			if mtl_model:
				name = task_name+"_p-"+str(prune)+"_MTLModel.pth"
			else:
				name = task_name+"_p-"+str(prune)+"_SingleTaskModel.pth"
			# model_object = load_model(name, folder)
			file_path = folder+"/"+name
			mean_lat, std_lat = get_lat(file_path, dataloader)
			inf_latency = mean_lat*1000
			model = model_info(name, accuracies, load_latency, inf_latency, tasks, file_path, prune)
			models.append(model)

			row = [i[0], i[1], mean_lat, std_lat, i[4], i[5], i[6]]
			with open("caliberated_," + model_file, 'a', newline='') as f:
			    writer = csv.writer(f, delimiter='\t')
			    writer.writerow(row)

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
			model.latency = round(float(model.latency), 2)
			latencies.add(model.latency)
		for latency in latencies:
			table[latency] = 0
		for model in models:
			model.latency = round(float(model.latency), 2)
			if(model.latency not in accuracy_table):
				accuracy_table[model.latency] = model.accuracy[i]
				table[model.latency] = model
			elif(accuracy_table[model.latency] < model.accuracy[i]):
				accuracy_table[model.latency] = model.accuracy[i]
				table[model.latency] = model

		task_tables[models[0].tasks[i]] = table
	return task_tables


def get_lat(model_path, dataloader):
    # Get latency
    latencies = []
    start = time.time()
    model = torch.load(model_path, map_location=torch.device('cpu'))
    load_time = time.time() - start
    # for i, sample in enumerate(eval_dataset):
    for i in range(100):
        start = time.time()
        model = model.cpu()
        # image = sample[0].unsqueeze(0)
        image = next(iter(dataloader))[0]
        output = model(image)
        lat = time.time() - start
        latencies.append(lat)
        if i >= 100:
            break
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)

    return load_time + mean_lat, std_lat