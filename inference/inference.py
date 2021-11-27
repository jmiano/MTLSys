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
		self.latency = latency
		self.tasks = tasks
		self.input_image = input_image
		

def process_request(request_details):
	model_to_use = None
	task_count = len(request_details.tasks)
	if(task_count > 1):
		for latency in three_task_table:
			if(latency <= request_details.latency):
				model_to_use = three_task_table[latency]
	else:
		if(request_details.tasks == "age"):
			for latency in age_task_table:
				if(latency <= request_details.latency):
					model_to_use = age_task_table[latency]	
		elif(request_details.tasks == "gender"):
			for latency in age_task_table:
				if(latency <= request_details.latency):
					model_to_use = gender_task_table[latency]	
		elif(request_details.tasks == "ethnicity"):	
			for latency in age_task_table:
				if(latency <= request_details.latency):
					model_to_use = ethnicity_task_table[latency]	

	if(model_to_use == None):
		## Cannot meet latency requirement with any model
		return None, None
	else:
		output = model_to_use.model(request_details.input_image)
		return output, model_to_use.accuracy

def process_request_with_accuracy(request_details):
	model_to_use = None
	task_count = len(request_details.tasks)
	if(task_count > 1):
		for latency in three_task_table:
			if(latency <= request_details.latency and three_task_table[latency].accuracy >= request_details.accuracy):
				model_to_use = three_task_table[latency]
	else:
		if(request_details.tasks == "age"):
			for latency in age_task_table:
				if(latency <= request_details.latency and age_task_table[latency].accuracy >= request_details.accuracy):
					model_to_use = age_task_table[latency]	
		elif(request_details.tasks == "gender"):
			for latency in age_task_table:
				if(latency <= request_details.latency and gender_task_table[latency].accuracy >= request_details.accuracy):
					model_to_use = gender_task_table[latency]	
		elif(request_details.tasks == "ethnicity"):	
			for latency in age_task_table:
				if(latency <= request_details.latency and ethnicity_task_table[latency].accuracy >= request_details.accuracy):
					model_to_use = ethnicity_task_table[latency]	

	if(model_to_use == None):
		## Cannot meet latency requirement with any model
		return None, None
	else:
		output = model_to_use.model(request_details.input_image)
		return output, model_to_use.accuracy


def process_batch_requests(requests, min_accuracy=False):

	## Can write some queing stuff
	## iterate on the requests and call process_request
	latency_hit = []
	accuracy_hit = []
	for request in requests:
		if(min_accuracy):
			start_time = time.time()
			output, accuracy = process_request_with_accuracy(request)
			end_time = time.time()
		else:
			start_time = time.time()
			output, accuracy = process_request(request)
			end_time = time.time()
			if(end_time-start_time <= request.latency):
				latency_hit.append(1)
			else:
				latency_hit.append(0)
	return latency_hit, accuracy_hit

## Prepare inference testing data 

### Load in the data
folder = '../UTKFace'
transform = data_transform()
dataset = FacesDataset(folder=folder, transform=transform)

train_len = int(len(dataset)*0.8)
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len], torch.Generator().manual_seed(8))


train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)


## Prepare models for inference

## MTL Pruned Network System

### All three tasks
three_task_models = load_all_task_models_info("model_score_lookup_multitask.tsv ", "../models/model_variants")
three_task_table = get_models_table(three_task_models)

### Age
age_task_models = load_one_task_models_info("model_score_lookup_multitask.tsv", "../models/model_variants" "age", True)
age_task_table = get_models_table(age_task_models)

### Gender
gender_task_models = load_one_task_models_info("model_score_lookup_multitask.tsv ", "../models/model_variants", "gender", True)
gender_task_table = get_models_table(gender_task_models)

### Ethnicity
ethnicity_task_models = load_one_task_models_info("model_score_lookup_multitask.tsv ", "../models/model_variants/", "ethnicity", True)
ethnicity_task_table = get_models_table(ethnicity_task_models)


## Run system on age tasks | MTL Network

min_latency = 0.35
max_latency = 0.95
batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = "age"
hits = []
for i in range(min_latency, max_latency, 0.1):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit = process_batch_requests(requests)
		requests = []
		hits+=lat_hit

	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit

SLO_hit = calculate_SLO_hit(hits)

print("Age Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))


## Run system on gender tasks | MTL Network

min_latency = 0.32
max_latency = 1.10
batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = "gender"
hits = []
for i in range(min_latency, max_latency, 0.1):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit = process_batch_requests(requests)
		requests = []
		hits+=lat_hit

	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit

SLO_hit = calculate_SLO_hit(hits)

print("Gender Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))

## Run system on ethnicity tasks | MTL Network

min_latency = 0.32
max_latency = 1.02
batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = "ethnicity"
hits = []
for i in range(min_latency, max_latency, 0.1):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit = process_batch_requests(requests)
		requests = []
		hits+=lat_hit

	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit

SLO_hit = calculate_SLO_hit(hits)

print("Ethnicity Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))


## Run system on all tasks | MTL Network

min_latency = 0.33
max_latency = 0.95
batch_size = 8 # No relevance now since we are not doing queueing
requests = []
tasks = ["age", "gender", "ethnicity"]
hits = []
for i in range(min_latency, max_latency, 0.1):
	if(len(requests) == batch_size):	
		lat_hit, acc_hit = process_batch_requests(requests)
		requests = []
		hits+=lat_hit

	request = RequestDetails(0, i, tasks, next(iter(dataloader))[0])
	requests.append(request)

lat_hit, acc_hit = process_batch_requests(requests)
hits+=lat_hit

SLO_hit = calculate_SLO_hit(hits)

print("All Task MTL System (No Accuracy) SLO hit/mishit ratio: " + str(SLO_hit) + ", hits: " + str(sum(hits)) + ", mishits: " + str(len(hits) - sum(hits)))

