use_gpu = False
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
					mean_acc = (float(three_task_table["age"][latency].accuracy[0]) + float(three_task_table["gender"][latency].accuracy[1]) + float(three_task_table["ethnicity"][latency].accuracy[2])) / 3.0
					mean_acc_existing = (float(model_to_use.accuracy[0]) + float(model_to_use.accuracy[1]) + float(model_to_use.accuracy[2]))/3.0 
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
						if(model_to_use.accuracy[0] < float(gender_task_table["gender"][latency].accuracy[0])):
							model_to_use = gender_task_table["gender"][latency]
		elif(request_details.tasks[0] == "ethnicity"):	
			for latency in ethnicity_task_table["ethnicity"]:
				if(latency <= request_details.latency):
					if(model_to_use == None):
						model_to_use = ethnicity_task_table["ethnicity"][latency]	
					else:
						if(model_to_use.accuracy[0] < float(ethnicity_task_table["ethnicity"][latency].accuracy[0])):
							model_to_use = ethnicity_task_table["ethnicity"][latency]

	if(model_to_use == None):
		## Cannot meet latency requirement with any model
		return None, None
	else:
		model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
		# model = torch.load(model_to_use.file_path)
		# model.cuda()
		model.cpu()
		# output = model(request_details.input_image.cuda())
		return output, model_to_use.latency
		if(task_count > 1):
			return output, (float(model_to_use.accuracy[0]) + float(model_to_use.accuracy[1]) + float(model_to_use.accuracy[2]))/3.0
		else:
			return output, model_to_use.accuracy[0]

def process_request_with_accuracy(request_details):
	model_to_use = None
	task_count = len(request_details.tasks)
	if(task_count > 1):
		for latency in three_task_table["age"]:
			# print(float(three_task_table["gender"][latency].accuracy[1]), request_details.accuracy[1])
			mean_acc = float(three_task_table["age"][latency].accuracy[0]) >= request_details.accuracy[0] and\
				float(three_task_table["gender"][latency].accuracy[1]) >= request_details.accuracy[1] and\
				float(three_task_table["ethnicity"][latency].accuracy[2]) >= request_details.accuracy[2]
			if(latency <= request_details.latency and mean_acc == True):
					model_to_use = three_task_table["age"][latency]
	else:
		if(request_details.tasks[0] == "age"):
			for latency in age_task_table["age"]:
				if(latency <= request_details.latency and float(age_task_table["age"][latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = age_task_table["age"][latency]	
					else:
						if(float(model_to_use.accuracy[0]) < float(age_task_table["age"][latency].accuracy[0])):
							model_to_use = age_task_table["age"][latency]
		elif(request_details.tasks[0] == "gender"):
			for latency in gender_task_table["gender"]:
				if(latency <= request_details.latency and float(gender_task_table["gender"][latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = gender_task_table["gender"][latency]	
					else:
						if(float(model_to_use.accuracy[0]) < float(gender_task_table["gender"][latency].accuracy[0])):
							model_to_use = gender_task_table["gender"][latency]
		elif(request_details.tasks[0] == "ethnicity"):	
			for latency in ethnicity_task_table["ethnicity"]:
				if(latency <= request_details.latency and float(ethnicity_task_table["ethnicity"][latency].accuracy[0]) >= request_details.accuracy):
					if(model_to_use == None):
						model_to_use = ethnicity_task_table["ethnicity"][latency]	
					else:
						if(float(model_to_use.accuracy[0]) < float(ethnicity_task_table["ethnicity"][latency].accuracy[0])):
							model_to_use = ethnicity_task_table["ethnicity"][latency]

	if(model_to_use == None):
		## Cannot meet latency/accuracy requirement with any model
		return None, None
	else:
		# print("asd", request_details.tasks)
		# print(model_to_use.latency, end=" ")
		model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
		# model = torch.load(model_to_use.file_path)
		# model.cuda()
		model.cpu()
		# output = model(request_details.input_image.cuda())
		output = model(request_details.input_image)
		return output, model_to_use.accuracy[0]
		# return output, model_to_use.latency

def process_request_only_mtl(request_details, with_accuracy=False):
	if(request_details.tasks[0] == "age"):
		model_to_use = None
		for latency in three_task_table["age"]:
			if(latency <= request_details.latency):
				if(with_accuracy):
					if( float(three_task_table["age"][latency].accuracy[0]) < request_details.accuracy): continue
				if(model_to_use == None):
					model_to_use = three_task_table["age"][latency]
				else:
					if(float(model_to_use.accuracy[0]) < float(three_task_table["age"][latency].accuracy[0])):
						model_to_use = three_task_table["age"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			# output = model(request_details.input_image.cuda())
			output = model(request_details.input_image)
			return model_to_use.latency, model_to_use.accuracy[0]
	elif(request_details.tasks[0] == "gender"):
		model_to_use = None
		for latency in three_task_table["gender"]:
			if(latency <= request_details.latency):
				if(with_accuracy):
					if( float(three_task_table["gender"][latency].accuracy[1]) < request_details.accuracy): continue
				if(model_to_use == None):
					model_to_use = three_task_table["gender"][latency]	
				else:
					if(float(model_to_use.accuracy[1]) < float(three_task_table["gender"][latency].accuracy[1])):
						model_to_use = three_task_table["gender"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			output = model(request_details.input_image)
			# output = model(request_details.input_image.cuda())
			return model_to_use.latency, model_to_use.accuracy[1]

	elif(request_details.tasks[0] == "ethnicity"):	
		model_to_use = None
		for latency in three_task_table["ethnicity"]:
			if(latency <= request_details.latency):
				if(with_accuracy):
					if( float(three_task_table["ethnicity"][latency].accuracy[2]) < request_details.accuracy): continue
				if(model_to_use == None):
					model_to_use = three_task_table["ethnicity"][latency]	
				else:
					if(float(model_to_use.accuracy[2]) < float(three_task_table["ethnicity"][latency].accuracy[2])):
						model_to_use = three_task_table["ethnicity"][latency]
		if(model_to_use == None):
			return None, None
		else:
			model = torch.load(model_to_use.file_path, map_location=torch.device('cpu'))
			model.cpu()
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			# output = model(request_details.input_image.cuda())
			output = model(request_details.input_image)
			return model_to_use.latency, model_to_use.accuracy[2]


def process_request_single_model_system(request_details, with_accuracy=False):
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
				if(with_accuracy):
					if(len(request_details.tasks) > 1): acc = request_details.accuracy[0]
					else: acc = request_details.accuracy
					if( float(single_age_task_table["age"][latency].accuracy[0]) < acc): continue
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
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			# output = model(request_details.input_image.cuda())
			output = model(request_details.input_image)
			perf.append(model_to_use.accuracy[0])
	if(len(request_details.tasks) > 1 or request_details.tasks[0] == "gender"):
		model_to_use = None
		for latency in single_gender_task_table["gender"]:
			if(latency <= req_latency):
				if(with_accuracy):
					if(len(request_details.tasks) > 1): acc = request_details.accuracy[1]
					else: acc = request_details.accuracy
					if( float(single_gender_task_table["gender"][latency].accuracy[0]) < acc): continue
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
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			output = model(request_details.input_image)
			# output = model(request_details.input_image.cuda())
			perf.append(model_to_use.accuracy[0])
	if(len(request_details.tasks) > 1 or request_details.tasks[0] == "ethnicity"):	
		model_to_use = None
		for latency in single_ethnicity_task_table["ethnicity"]:
			if(latency <= req_latency):
				if(with_accuracy):
					if(len(request_details.tasks) > 1): acc = request_details.accuracy[2]
					else: acc = request_details.accuracy
					if( float(single_ethnicity_task_table["ethnicity"][latency].accuracy[0]) < acc): continue
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
			# model = torch.load(model_to_use.file_path)
			# model.cuda()
			output = model(request_details.input_image)
			# output = model(request_details.input_image.cuda())
			perf.append(model_to_use.accuracy[0])

	return model_to_use.latency, np.mean(perf)

def process_batch_requests(requests, min_accuracy=False, mtl_model=True, only_mtl=False):
	use_gpu = False
	## Can write some queing stuff
	## iterate on the requests and call process_request
	latency_hit = []
	accuracy_hit = []
	lat_acc_pair = []
	outputs, latencies = [], []
	for request in requests:
		if(min_accuracy):
			if use_gpu: # skip first sample (to avoid slow GPU processing on first sample)
				if(only_mtl):
					output, accuracy = process_request_only_mtl(request, with_accuracy=True)
				elif(mtl_model):
					output, accuracy = process_request_with_accuracy(request)
				else:                
					output, accuracy = process_request_single_model_system(request, with_accuracy=True)
                    
			start_time = time.time()
			if(only_mtl):
				output, accuracy = process_request_only_mtl(request, with_accuracy=True)
			elif(mtl_model):
				output, accuracy = process_request_with_accuracy(request)
			else:                
				output, accuracy = process_request_single_model_system(request, with_accuracy=True)
			end_time = time.time()
			inf_time = (end_time-start_time)*1000
			# print(inf_time, request.latency, accuracy)
			# print(inf_time, start_time, end_time)
			# print(inf_time)
			outputs.append(output)
			latencies.append(inf_time)
			if(inf_time <= request.latency + 5 and output!=None):
				latency_hit.append(1)
			else:
				latency_hit.append(0)

			if(accuracy!=None ):
				accuracy_hit.append(1)
			else:
				accuracy_hit.append(0)
			
		else:
			if use_gpu: # skip first sample (to avoid slow GPU processing on first sample)
				if(only_mtl):
					output, accuracy = process_request_only_mtl(request)
				elif(mtl_model):
					output, accuracy = process_request(request)
				else:
					output, accuracy = process_request_single_model_system(request)                
			start_time = time.time()
			if(only_mtl):
				output, accuracy = process_request_only_mtl(request)
			elif(mtl_model):
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
	return outputs, latencies
	# return latency_hit, accuracy_hit, lat_acc_pair