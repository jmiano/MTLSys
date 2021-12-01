from util import *
from eval import *
from data import FacesDataset, data_transform
import time
# from ..utils.data import FacesDataset, data_transform
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

class RequestDetails(object):
	"""docstring for RequestDetails"""
	def __init__(self, accuracy, latency, tasks, input_image):
		self.accuracy = accuracy
		self.latency = int(latency)
		self.tasks = tasks
		self.input_image = input_image
		

def process_request(request_details):
	model_to_use = None
	# print(three_task_table)
	task_count = len(request_details.tasks)
	if(task_count > 1):
		for latency in three_task_table["age"]:
			if(latency <= request_details.latency):
				if(model_to_use == None):
					model_to_use = three_task_table["age"][latency]
				else:	
					mean_acc = (float(three_task_table["age"][latency].accuracy[0]) + float(three_task_table["gender"][latency].accuracy[0]) + float(three_task_table["ethnicity"][latency].accuracy[0])) / 3.0
					mean_acc_existing = sum(model_to_use.accuracy)/3.0 
					if(mean_acc_existing < mean_acc): model_to_use = three_task_table["age"][latency] 
	else:
		if(request_details.tasks[0] == "age"):
			for latency in age_task_table["age"]:
				if(latency <= request_details.latency):
					if(model_to_use == None):
						model_to_use = age_task_table["age"][latency]
					else:
						if(model_to_use.accuracy[0] < float(age_task_table["age"][latency].accuracy[0])):
							model_to_use = age_task_table["age"][latency]
		elif(request_details.tasks[0] == "gender"):
			for latency in gender_task_table["gender"]:
				if(latency <= request_details.latency):
					if(model_to_use == None):
						model_to_use = gender_task_table["gender"][latency]	
					else:
						if(model_to_use.accuracy[0] < float(age_task_table["gender"][latency].accuracy[0])):
							model_to_use = gender_task_table["gender"][latency]
		elif(request_details.tasks[0] == "ethnicity"):	
			for latency in ethnicity_task_table["ethnicity"]:
				if(latency <= request_details.latency):
					if(model_to_use == None):
						model_to_use = ethnicity_task_table["ethnicity"][latency]	
					else:
						if(model_to_use.accuracy[0] < float(age_task_table["ethnicity"][latency].accuracy[0])):
							model_to_use = ethnicity_task_table["ethnicity"][latency]

	if(model_to_use == None):
		## Cannot meet latency requirement with any model
		return None, None
	else:
		model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
		model.cpu()
		output = model(request_details.input_image)
		# return output, model_to_use.latency
		if(task_count > 1):
			return output, model_to_use.accuracy[0]
		else:
			return output, sum(model_to_use.accuracy)/3.0

def process_request_with_accuracy(request_details):
	model_to_use = None
	task_count = len(request_details.tasks)
	if(task_count > 1):
		for latency in three_task_table["age"]:
			mean_acc = float(three_task_table["age"][latency].accuracy[0]) + float(three_task_table["gender"][latency].accuracy[0]) + float(three_task_table["ethnicity"][latency].accuracy[0])
			if(latency <= request_details.latency and float(mean_acc) >= request_details.accuracy):
				if (model_to_use == None):
					model_to_use = three_task_table["age"][latency]
				else:
					mean_acc_existing = sum(model_to_use.accuracy)/3.0 
					if(mean_acc_existing < mean_acc): model_to_use = three_task_table["age"][latency] 

	else:
		if(request_details.tasks[0] == "age"):
			for latency in age_task_table["age"]:
				if(latency <= request_details.latency and float(age_task_table["age"][latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = age_task_table["age"][latency]	
					else:
						if(model_to_use.accuracy[0] < float(three_task_table["age"][latency].accuracy[0])):
							model_to_use = age_task_table["age"][latency]
		elif(request_details.tasks[0] == "gender"):
			for latency in gender_task_table["gender"]:
				if(latency <= request_details.latency and float(gender_task_table[latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = gender_task_table[latency]	
					else:
						if(model_to_use.accuracy[0] < float(three_task_table["gender"][latency].accuracy[0])):
							model_to_use = gender_task_table["gender"][latency]
		elif(request_details.tasks[0] == "ethnicity"):	
			for latency in ethnicity_task_table["ethnicity"]:
				if(latency <= request_details.latency and float(ethnicity_task_table[latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = ethnicity_task_table[latency]	
					else:
						if(model_to_use.accuracy[0] < float(three_task_table["ethnicity"][latency].accuracy[0])):
							model_to_use = ethnicity_task_table["ethnicity"][latency]

	if(model_to_use == None):
		## Cannot meet latency/accuracy requirement with any model
		return None, None
	else:
		# print("asd", request_details.tasks)
		# print(request_details.accuracy, model_to_use.accuracy)
		model = torch.load(model_to_use.file_path)
		model.cpu()
		output = model(request_details.input_image)
		return output, model_to_use.accuracy[0]
		# return output, model_to_use.latency

def process_request_only_mtl(request):
	if(request_details.tasks[0] == "age"):
		model_to_use = None
		for latency in three_task_table["age"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = three_task_table["age"][latency]
				else:
					if(model_to_use.accuracy[0] < float(three_task_table["age"][latency].accuracy[0])):
						model_to_use = three_task_table["age"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			return output, model_to_use.accuracy[0]
	elif(request_details.tasks[0] == "gender"):
		model_to_use = None
		for latency in three_task_table["gender"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = three_task_table["gender"][latency]	
				else:
					if(model_to_use.accuracy[0] < float(three_task_table["gender"][latency].accuracy[0])):
						model_to_use = three_task_table["gender"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			return output, model_to_use.accuracy[1]

	elif(request_details.tasks[0] == "ethnicity"):	
		model_to_use = None
		for latency in three_task_table["ethnicity"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = three_task_table["ethnicity"][latency]	
				else:
					if(model_to_use.accuracy[0] < float(three_task_table["ethnicity"][latency].accuracy[0])):
						model_to_use = three_task_table["ethnicity"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			return output, model_to_use.accuracy[2]


def process_request_single_model_system(request):
	# print(three_task_table)
	task_count = len(request_details.tasks)
	if( len(request_details.tasks) > 1): req_latency = request_details.latency/3.0
	else: req_latency = request_details.latency 
	perf = []
	output = None
	if(len(request_details.tasks) > 1 or request_details.tasks[0] == "age"):
		model_to_use = None
		for latency in single_age_task_table["age"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = single_age_task_table["age"][latency]
				else:
					if(model_to_use.accuracy[0] < float(single_age_task_table["age"][latency].accuracy[0])):
						model_to_use = single_age_task_table["age"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			perf.append(model_to_use.accuracy[0])
	if(len(request_details.tasks) > 1 or request_details.tasks[0] == "gender"):
		model_to_use = None
		for latency in single_gender_task_table["gender"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = single_gender_task_table["gender"][latency]	
				else:
					if(model_to_use.accuracy[0] < float(single_gender_task_table["gender"][latency].accuracy[0])):
						model_to_use = single_gender_task_table["gender"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			perf.append(model_to_use.accuracy[0])
	if(len(request_details.tasks) > 1 or request_details.tasks[0] == "ethnicity"):	
		model_to_use = None
		for latency in single_ethnicity_task_table["ethnicity"]:
			if(latency <= req_latency):
				if(model_to_use == None):
					model_to_use = single_ethnicity_task_table["ethnicity"][latency]	
				else:
					if(model_to_use.accuracy[0] < float(single_ethnicity_task_table["ethnicity"][latency].accuracy[0])):
						model_to_use = single_ethnicity_task_table["ethnicity"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			output = model(request_details.input_image)
			perf.append(model_to_use.accuracy[0])

	return output, np.mean(perf)

def process_batch_requests(requests, min_accuracy=False, mtl_model=True):

	## Can write some queing stuff
	## iterate on the requests and call process_request
	latency_hit = []
	accuracy_hit = []
	lat_acc_pair = []
	for request in requests:
		if(min_accuracy):
			start_time = time.time()
			output, accuracy = process_request_with_accuracy(request)
			end_time = time.time()
			inf_time = (end_time-start_time)*1000
			# print(inf_time, request.latency, accuracy)
			# print(inf_time, start_time, end_time)
			if(inf_time <= request.latency and output!=None):
				latency_hit.append(1)
			else:
				latency_hit.append(0)

			if(accuracy!=None and accuracy >= request.accuracy ):
				accuracy_hit.append(1)
			else:
				accuracy_hit.append(0)
			
		else:
			start_time = time.time()
			if(mtl_model):
				output, accuracy = process_request(request)
			else:
				output, accuracy = process_request_single_model_system(request)
			end_time = time.time()
			inf_time = (end_time-start_time)*1000
			# print(inf_time, request.latency, accuracy)
			# print(inf_time, end_time, output)
			if(inf_time <= request.latency and output!=None):
				latency_hit.append(1)
			else:
				latency_hit.append(0)
			if (accuracy != None):
				lat_acc_pair.append([request.latency, accuracy])
	return latency_hit, accuracy_hit, lat_acc_pair

def plot_double_pareto_curve_graph(pair1, pair2):
	fig, ax = plt.subplots(1, figsize=(8, 6))
	fig.suptitle('Multiple Lines in Same Plot', fontsize=15)

	x1, y1 = []
	x2, y2 = []
	for i in pair1:
		x1.append(i[0])
		y1.append(i[1])
	for i in pair2:
		x2.append(i[0])
		y2.append(i[1])

	ax.plot(x1, y1, color="red", label="My Line 1")
	ax.plot(x2, y2, color="green", label="My Line 2")

	plt.legend(loc="lower right", title="Legend Title", frameon=False)

	plt.show()
## Prepare inference testing data 

### Load in the data
folder = '../UTKFace'
transform = data_transform()
dataset = FacesDataset(folder=folder, transform=transform)

train_len = int(len(dataset)*0.8)
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len], torch.Generator().manual_seed(8))


dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

## Prepare models for inference

## MTL Pruned Network System

print("Loading all task models..")
### All three tasks
three_task_models = load_all_task_models_info(dataloader, "model_score_lookup_multitask.tsv", "../models/model_variants")
three_task_table = get_models_table(three_task_models)

# print("Loading age models..")
# ### Age
# age_task_models = load_one_task_models_info(dataloader, "model_score_lookup_multitask.tsv", "../models/model_variants", "age", True)
# age_task_table = get_models_table(age_task_models)
# print("Loading gender models..")
# ### Gender
# gender_task_models = load_one_task_models_info(dataloader, "model_score_lookup_multitask.tsv", "../models/model_variants", "gender", True)
# gender_task_table = get_models_table(gender_task_models)

# print("Loading ethnicity models..")
# ### Ethnicity
# ethnicity_task_models = load_one_task_models_info(dataloader, "model_score_lookup_multitask.tsv", "../models/model_variants/", "ethnicity", True)
# ethnicity_task_table = get_models_table(ethnicity_task_models)

print("Loading single age models..")
### Age
single_age_task_models = load_one_task_models_info(dataloader, "model_score_lookup_singletask.tsv", "../models/model_variants", "age", True)
single_age_task_table = get_models_table(single_age_task_models)
print("Loading single gender models..")
### Gender
single_gender_task_models = load_one_task_models_info(dataloader, "model_score_lookup_singletask.tsv", "../models/model_variants", "gender", True)
single_gender_task_table = get_models_table(single_gender_task_models)

print("Loading single ethnicity models..")
### Ethnicity
single_ethnicity_task_models = load_one_task_models_info(dataloader, "model_score_lookup_singletask.tsv", "../models/model_variants/", "ethnicity", True)
single_ethnicity_task_table = get_models_table(single_ethnicity_task_models)


min_latency = int(min(three_task_table["age"]))
max_latency = int(max(three_task_table["age"])) + 10

## Run system on age tasks | MTL Network


# batch_size = 8 # No relevance now since we are not doing queueing
# requests = []
# tasks = ["age"]
# hits = []
# for i in range(min_latency, max_latency):
# 	if(len(requests) == batch_size):	
# 		lat_hit, acc_hit = process_batch_requests(requests)
# 		requests = []
# 		hits+=lat_hit

# 	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
# 	requests.append(request)

# lat_hit, acc_hit = process_batch_requests(requests)
# hits+=lat_hit

# SLO_hit = calculate_SLO_hit(hits)

# print("Age Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))
# print()

# requests = []
# hits = []
# acc_hits = []
# accuracy = 0.75
# for i in range(min_latency, max_latency):
# 	if(len(requests) == batch_size):	
# 		lat_hit, acc_hit = process_batch_requests(requests, min_accuracy=True)
# 		requests = []
# 		hits+=lat_hit
# 		acc_hits+=acc_hit
# 	request = RequestDetails(accuracy, i, tasks, next(iter(dataloader))[0])
# 	requests.append(request)

# lat_hit, acc_hit = process_batch_requests(requests, min_accuracy=True)
# hits+=lat_hit
# acc_hits+=acc_hit

# SLO_hit = calculate_SLO_hit(hits)
# Acc_hit = calculate_SLO_hit(acc_hits)

# print("Age Task MTL System with 75% SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))
# print("75% Accuracy hit/mishit ratio: " + str(Acc_hit) + ", hits: " + str(sum(acc_hits)) + ", mishits: " + str(len(acc_hits) - sum(acc_hits)))
# print()

# ## Run system on gender tasks | MTL Network

# # min_latency = 32
# # max_latency = 110
# batch_size = 8 # No relevance now since we are not doing queueing
# requests = []
# tasks = ["gender"]
# hits = []
# for i in range(min_latency, max_latency):
# 	if(len(requests) == batch_size):	
# 		lat_hit, acc_hit = process_batch_requests(requests)
# 		requests = []
# 		hits+=lat_hit

# 	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
# 	requests.append(request)

# lat_hit, acc_hit = process_batch_requests(requests)
# hits+=lat_hit

# SLO_hit = calculate_SLO_hit(hits)

# print("Gender Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))
# print()

# ## Run system on ethnicity tasks | MTL Network

# # min_latency = 32
# # max_latency = 102
# batch_size = 8 # No relevance now since we are not doing queueing
# requests = []
# tasks = ["ethnicity"]
# hits = []
# for i in range(min_latency, max_latency):
# 	if(len(requests) == batch_size):	
# 		lat_hit, acc_hit = process_batch_requests(requests)
# 		requests = []
# 		hits+=lat_hit

# 	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
# 	requests.append(request)

# lat_hit, acc_hit = process_batch_requests(requests)
# hits+=lat_hit

# SLO_hit = calculate_SLO_hit(hits)

# print("Ethnicity Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))
# print()


## Run system on all tasks | MTL Network

# min_latency = 33
# max_latency = 95
batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = ["age", "gender", "ethnicity"]
hits = []
lat_acc = []
for i in range(min_latency, max_latency):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit, lat_acc_pair = process_batch_requests(requests)
		requests = []
		hits+=lat_hit
		lat_acc+=lat_acc_pair
	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit
lat_acc+=lat_acc_pair

# SLO_hit = calculate_SLO_hit(hits)

# print("All Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))
# print()


batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = ["age", "gender", "ethnicity"]
hits = []
lat_acc1 = []
for i in range(min_latency, max_latency):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit, lat_acc_pair = process_batch_requests(requests, False, False)
		requests = []
		hits+=lat_hit
		lat_acc1+=lat_acc_pair
	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit
lat_acc1+=lat_acc_pair


plot_double_pareto_curve_graph(lat_acc, lat_acc1)